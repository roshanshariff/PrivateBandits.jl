module Experiments

using LinearAlgebra
using Statistics
using Random
using Distributions: Beta, Truncated
using PDMats: colwise_sumsq!
using Parameters
using RecipesBase

using Distributed: pmap
import ProgressMeter
include("PmapProgressMeter.jl")

using ..DifferentialPrivacy
using ..LinearBandits
using ..Accumulators


export GapArms, SubspaceArms, make_alg, make_strategy, run_episode,
    ExpResult, pmap_progress, run_experiment

#=========================================================================#

# Rademacher arms

struct GapArms
    opt :: Float64
    maxsubopt :: Float64
    minsubopt :: Float64
    norm :: Float64
    function GapArms(env::EnvParams; gap=0.0)
        opt = env.maxrewardmean/(env.maxθnorm*env.maxactionnorm)
        new(opt, opt-gap, -opt, env.maxactionnorm)
    end
end

function (f::GapArms)(arms::AbstractMatrix)
    (d, k) = size(arms)
    β = (d - 1)/2  # https://stats.stackexchange.com/a/85977
    dist = Truncated(Beta(β, β), (1+f.minsubopt)/2, (1+f.maxsubopt)/2)
    heads = @view arms[1:1, :]
    tails = @view arms[2:end, :]
    randn!(tails)
    colwise_sumsq!(heads, tails, 1)
    tails ./= .√heads
    rand!(dist, heads)
    heads .= f.norm .* (2 .* heads .- 1)
    @inbounds heads[rand(1:k)] = f.norm * f.opt
    tails .*= sqrt.(f.norm^2 .- heads.^2)
    arms
end

struct SubspaceArms{T}
    arms :: T
    subspacedims :: Rational{Int}
    norm :: Float64
    function SubspaceArms(env::EnvParams, arms, subspacedims) where {}
        new{typeof(arms)}(arms, subspacedims, env.maxactionnorm)
    end
end

function (f::SubspaceArms)(arms::AbstractMatrix)
    (D, k) = size(arms)
    d = ceil(Int, D*f.subspacedims)

    top = @view arms[1:d, :]
    bottom = @view arms[d+1:end, :]
    f.arms(top)
    mul!(bottom, randn(D-d, d), top)

    sqnorms = colwise_sumsq!(zeros(1, k), arms, 1)
    arms .*= f.norm ./ .√sqnorms
end


#=========================================================================#

@with_kw struct GaussianReward
    σ :: Float64
end

(dist::GaussianReward)(mean) = mean + dist.σ*randn()

subgaussian_σ(dist::GaussianReward) = dist.σ

@with_kw struct BoundedReward @deftype Float64
    min = -1.0
    max = +1.0
    @assert isfinite(min) && isfinite(max)
    @assert min ≤ max
end

function (dist::BoundedReward)(mean)
    dist.min ≤ mean ≤ dist.max ||
        throw(ArgumentError("reward $mean ∉ [$(dist.min), $(dist.max)]"))
    u = dist.min + rand()*(dist.max - dist.min)
    mean ≤ u ? dist.min : dist.max
end

# http://blog.wouterkoolen.info/BernoulliSubGaussian/post.html
subgaussian_σ(dist::BoundedReward) = (dist.max - dist.min)/2

function reward_noise(env::EnvParams)
    if isfinite(env.maxreward)
        noise = BoundedReward(min=-env.maxreward, max=env.maxreward)
        (EnvParams(env; σ=subgaussian_σ(noise)), noise)
    else
        noise = GaussianReward(σ=env.σ)
        (env, noise)
    end
end

function make_alg(env::EnvParams, horizon::Integer; ρ=1.0)
    reg = RegParams(ρmin=ρ, ρmax=ρ, γ=0, shift=ρ)
    EllipLinUCB(CholeskyUpdate(env.dim+1), env, reg, 1/horizon)
end

function make_alg(env::EnvParams, horizon::Integer, strategy::AccumStrategy)
    α = 1/2horizon
    EllipLinUCB(strategy, env, regparams(strategy; α=α), α)
end

function make_strategy(env::EnvParams, horizon::Integer, Mechanism,
                       Strategy=PanPrivTreeStrategy; ε, δ)
    L̃ = √(env.maxactionnorm^2 + env.maxreward^2)
    dp = DiffPrivParams(dim=env.dim+1, ε=ε, δ=δ, L̃=L̃)
    Strategy(SymMatrix(dp.dim), Mechanism, horizon, dp) :: AccumStrategy
end

function make_alg(env::EnvParams, horizon::Integer, args...; kwargs...)
    strategy = make_strategy(env, horizon, args...; kwargs...)
    make_alg(env, horizon, strategy)
end

#=========================================================================#

mutable struct Bandit{Alg<:ContextLinBandit, State}
    alg :: Alg
    state :: State
    function Bandit(alg::ContextLinBandit) where {}
        state = initialize(alg)
        new{typeof(alg), typeof(state)}(alg, state)
    end
end

function oneround!(b::Bandit, θ, arms, noise)
    i = choosearm(b.alg, b.state, arms)
    rewards = arms' * θ
    let arm = view(arms, :, i)
        b.state = learn!(b.alg, b.state, arm, noise(rewards[i]))
    end
    rewards .= maximum(rewards) .- rewards
    #println("Arm regrets = ", armregrets, ", chose arm ", chosenarm, ", got regret ", armregrets[chosenarm])
    rewards[i]
end

function run_episode(env::EnvParams, alg::ContextLinBandit, makearms,
                     horizon::Int;
                     numarms=env.dim^2, constarms=false, subsample=horizon)

    (env, noise) = reward_noise(env)
    θ = env.maxθnorm * normalize!(randn(env.dim))
    Q = let F = qr(θ)
        sign(F.R[]) * F.Q
    end

    alignedarms = zeros(env.dim, numarms)
    arms = zeros(env.dim, numarms)
    constarms && A_mul_B!(arms, Q, makearms(alignedarms))

    b = Bandit(alg)
    cumregret = 0.0
    skip = max(1, horizon ÷ subsample)
    regrets = [cumregret]
    for _ in skip:skip:horizon
        for _ in 1:skip
            constarms || mul!(arms, Q, makearms(alignedarms))
            cumregret += oneround!(b, θ, arms, noise)
        end
        push!(regrets, cumregret)
    end
    ExpResult(0:skip:horizon, regrets)
end

#=========================================================================#

struct ExpResult{X, Y}
    coords :: X
    data :: Y
end

@recipe function plot_ExpResult(result::ExpResult)
    x := result.coords
    _n = size(result.data, 2)
    _mean = mean(result.data, dims=2)
    _std = std(result.data; mean=_mean, dims=2) / √_n
    y := _mean
    ribbon := _std
    #marker --> true
    ()
end

function Base.hcat(results::ExpResult...)
    isempty(results) &&
        return ExpResult([], [])
    all(r -> r.coords == results[end].coords, results) ||
        throw(ArgumentError("Mismatch in x"))
    ExpResult(results[1].coords, reduce(hcat, (r.data for r in results)))
end

function Base.vcat(results::ExpResult...)
    isempty(results) && return ExpResult([], [])
    ExpResult(reduce(vcat, (r.coords for r in results)),
              reduce(vcat, (r.data for r in results)))
end

function pmap_progress(f, c, rest...; kwargs...)
    progress = ProgressMeter.Progress(length(c))
    pmap(f, progress, c, rest...; kwargs...)
end

function run_experiment(f, params; runs=1)
    experiment = (p for p in params, _ in 1:runs)
    ExpResult(collect(params), pmap_progress(f, experiment))
end

#=========================================================================#

end # module
