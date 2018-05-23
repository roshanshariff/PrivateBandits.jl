module Experiments

using Distributions: Beta, Truncated
using PDMats: colwise_sumsq!
using Parameters
using NamedTuples
using Plots

import ProgressMeter
import PmapProgressMeter

using DifferentialPrivacy
using LinearBandits
using Accumulators

export reward_noise, make_alg, make_private_alg, run_episode, orthoarms,
    unitarms, gaparms, constarms, ExpResult, pmap_progress, run_experiment

#=========================================================================#

# Rademacher arms

# function orthoarms(env::EnvParams)
#     arms = Matrix(Diagonal(fill(env.maxactionnorm, env.dim)))
#     () -> arms
# end

function gaparms(env::EnvParams; k=env.dim^2, gap=0.0)
    @unpack dim, maxactionnorm = env

    opt = env.maxrewardmean/(env.maxθnorm*maxactionnorm)
    maxsubopt = opt - gap
    minsubopt = -opt

    β = (dim-1)/2
    dist = Truncated(Beta(β, β), (1+minsubopt)/2, (1+maxsubopt)/2)

    arms = zeros(dim, k)
    heads = @view arms[1:1, :]
    tails = @view arms[2:end, :]

    function ()
        randn!(tails)
        colwise_sumsq!(heads, tails, 1)
        tails ./= sqrt.(heads)
        rand!(dist, heads)
        heads .= maxactionnorm .* (2.*heads .- 1)
        heads[rand(1:k)] = maxactionnorm * opt
        tails .*= sqrt.(maxactionnorm^2 .- heads.^2)
        arms
    end
end

function constarms(makearms)
    arms = makearms()
    () -> arms
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

function make_alg(env::EnvParams, horizon;
                  α=1/horizon, ρ=1.0, strategy=CholeskyUpdate)
    reg = RegParams(ρmin=ρ, ρmax=ρ, γ=0, shift=ρ)
    EllipLinUCB(strategy(env.dim+1), env, reg, α)
end

function make_private_alg(env, horizon, ε, δ, Mechanism;
                          α=1/horizon, strategy=PanPrivTreeStrategy)
    L̃ = √(env.maxactionnorm^2 + env.maxreward^2)
    dp = DiffPrivParams(dim=env.dim+1, ε=ε, δ=δ, L̃=L̃)
    s = strategy(SymMatrix(dp.dim), Mechanism, horizon, dp)
    EllipLinUCB(s, env, regparams(s, α/2), α/2)
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
    rewards = At_mul_B(arms, θ)
    let arm = view(arms, :, i)
        b.state = learn!(b.alg, b.state, arm, noise(rewards[i]))
    end
    rewards .= maximum(rewards) .- rewards
    #println("Arm regrets = ", armregrets, ", chose arm ", chosenarm, ", got regret ", armregrets[chosenarm])
    rewards[i]
end

function run_episode(env, alg, arms, noise, horizon;
                     randθ=false, subsample=1)
    b = Bandit(alg)
    θ = if randθ
        env.maxθnorm * normalize!(randn(dim(alg)))
    else
        [env.maxθnorm; zeros(dim(alg) - 1)]
    end
    cumregret = 0.0
    regrets = map(subsample:subsample:horizon) do _
        for i in 1:subsample
            cumregret += oneround!(b, θ, arms(), noise)
        end
        cumregret
    end
    ExpResult(subsample:subsample:horizon, regrets)
end

#=========================================================================#

struct ExpResult{X, Y}
    coords :: X
    data :: Y
end

@recipe function plot_ExpResult(result::ExpResult)
    x := result.coords
    _n = size(result.data, 2)
    _mean = mean(result.data, 2)
    _std = std(result.data, 2; mean=_mean) / √_n
    y := _mean
    ribbon := _std
    marker --> true
    ()
end

function Base.hcat(results::ExpResult...)
    if isempty(results)
        ExpResult([], [])
    else
        if !all(r -> r.coords == results[1].coords, results)
            throw(ArgumentError("Mismatch in x"))
        end
        ExpResult(results[1].coords, hcat((r.data for r in results)...))
    end
end

function Base.vcat(results::ExpResult...)
    ExpResult(vcat((r.coords for r in results)...),
              vcat((r.data for r in results)...))
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
