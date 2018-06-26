module DifferentialPrivacy

using LinearAlgebra
using Distributions
using Parameters
import Distributions
import PDMats
using Optim: optimize, minimizer

using ..LinearBandits: EnvParams, RegParams, regret_bound
using ..Accumulators

export TreeStrategy, PanPrivTreeStrategy, DiffPrivParams,
    GaussianMechanism, WishartMechanism, ShiftedWishart, shifted,
    OptShifted, regparams

abstract type ComposedStrategy <: AccumStrategy end

numcomposed(T::Type{<:ComposedStrategy}, horizon) = # function stub
    throw(MethodError(numcomposed, Tuple{Type{T}, typeof(horizon)}))

basestrategy(s::ComposedStrategy) = # function stub
    throw(MethodError(basestrategy, typeof((s,))))

noisedist(s::ComposedStrategy) = # function stub
    throw(MethodError(noisedist, typeof((s,))))

horizon(s::ComposedStrategy) = # function stub
    throw(MethodError(horizon, typeof((s,))))

numcomposed(s::ComposedStrategy) = numcomposed(typeof(s), horizon(s))

@with_kw struct DiffPrivParams @deftype Float64
    dim :: Int
    ε
    δ
    L̃
    @assert dim > 0
    @assert ε ≥ 0
    @assert δ ≥ 0
    @assert L̃ ≥ 0
    @assert isfinite(L̃)
end

function advanced_composition(m, dp::DiffPrivParams; δfail=dp.δ/2)
    @unpack ε, δ = dp
    DiffPrivParams(dp; ε = ε/√(8m*log(1/δfail)), δ = (δ-δfail)/m)
end

function (T::Type{<:ComposedStrategy})(s::AccumStrategy, Mechanism,
                                       horizon::Int, dp::DiffPrivParams)
    T(s, Mechanism(dp; m=numcomposed(T, horizon)), horizon)
end

regparams(s::ComposedStrategy; α, m=1) =
    regparams(noisedist(s); α=α/horizon(s), m=m*numcomposed(s))

#=========================================================================#
# Tree Strategy (not pan-private)

struct TreeAccum{State}
    count :: Int  # 0 ≤ count < 2^length(state)
    state :: Vector{State}
    TreeAccum(count::Int, state::Vector) where {} =
        new{eltype(state)}(count, state)
end

function TreeAccum(s::ComposedStrategy, init)
    initstate = accumulator(basestrategy(s), init)
    state = Vector{typeof(initstate)}(undef, numcomposed(s))
    state[end] = initstate
    TreeAccum(0, state)
end

struct TreeStrategy{S<:AccumStrategy, Noise} <: ComposedStrategy
    strategy :: S
    noise :: Noise
    horizon :: Int
    function TreeStrategy(s::AccumStrategy, noise, horizon) where {}
        horizon > 0 || throw(ArgumentError("Horizon $horizon ≤ 0"))
        new{typeof(s), typeof(noise)}(s, noise, nextpow2(n+1)-1)
    end
end

numcomposed(::Type{<:TreeStrategy}, horizon) = nbits(horizon)

basestrategy(s::TreeStrategy) = s.strategy

noisedist(s::TreeStrategy) = s.noise

horizon(s::TreeStrategy) = s.horizon

Accumulators.accumulator(s::TreeStrategy, init) = TreeAccum(s, init)

function Accumulators.accum!(s::TreeStrategy, a::TreeAccum, x)

    horizon = 1<<length(a.state) - 1
    active = (a.count & (horizon>>1)) ⊻ horizon
    initstate = a.state[1 + trailing_zeros(a.count | ~(horizon>>1))]

    for i in 1:trailing_ones(active)
        a.state[i] = accum(s.strategy, initstate, s.noise())
    end

    for i in IntBits(active)
        a.state[i] = accum!(s.strategy, a.state[i], x)
    end

    TreeAccum((a.count + 1) & horizon, a.state)
end

function Accumulators.accumulated(s::TreeStrategy, a::TreeAccum)
    horizon = 1<<length(a.state) - 1
    leafpos = 1 + trailing_zeros(a.count | ~(horizon>>1))
    accumulated(s.strategy, a.state[leafpos])
end

#=========================================================================#
# Pan-Private Tree Strategy

struct PanPrivTreeStrategy{S<:AccumStrategy, Noise} <: ComposedStrategy
    strategy :: S
    noise :: Noise
    horizon :: Int
    function PanPrivTreeStrategy(s::AccumStrategy, noise, horizon::Int) where {}
        horizon > 0 || throw(ArgumentError("Horizon $horizon ≤ 0"))
        new{typeof(s), typeof(noise)}(s, noise, nextpow(2, horizon))
    end
end

numcomposed(::Type{<:PanPrivTreeStrategy}, horizon) = 1 + nbits(horizon-1)

basestrategy(s::PanPrivTreeStrategy) = s.strategy

noisedist(s::PanPrivTreeStrategy) = s.noise

horizon(s::PanPrivTreeStrategy) = s.horizon

Accumulators.accumulator(s::PanPrivTreeStrategy, init) = TreeAccum(s, init)

function Accumulators.accum!(s::PanPrivTreeStrategy, a::TreeAccum, x)

    mask = 1<<(length(a.state) - 1) - 1
    leafpos = 1 + count_ones(a.count)
    newaccum = leafpos + trailing_zeros(a.count | ~mask)

    for i = length(a.state):-1:newaccum
        a.state[i] = accum!(s.strategy, a.state[i], x)
    end

    a.state[newaccum] = accum!(s.strategy, a.state[newaccum], s.noise())

    for i = newaccum-1:-1:leafpos
        a.state[i] = accum(s.strategy, a.state[i+1], s.noise())
    end

    TreeAccum((a.count + 1) & mask, a.state)
end

function Accumulators.accumulated(s::PanPrivTreeStrategy, a::TreeAccum)
    mask = 1<<(length(a.state) - 1) - 1
    prevcount = (a.count - 1) & mask
    leafpos = 1 + count_ones(prevcount)
    accumulated(s.strategy, a.state[leafpos])
end

#=========================================================================#
# Gaussian mechanism

struct GaussianMechanism
    dim :: Int
    σ :: Float64
    GaussianMechanism(dp::DiffPrivParams; m=1) =
        new(dp.dim, √(16m) * dp.L̃^2 * log(4/dp.δ) / dp.ε)
end

function (noise::GaussianMechanism)()
    M = zeros(noise.dim, noise.dim)
    for j = 1:noise.dim
        for i = 1:j-1
            @inbounds M[i,j] = noise.σ*randn()
        end
        @inbounds M[j,j] = √2*noise.σ*randn()
    end
    Symmetric(M, :U)
end

function regparams(mechanism::GaussianMechanism; α, m=1)
    d = mechanism.dim - 1
    σmax = mechanism.σ * √(2m) * (√(16d) - 2log(α))
    γ = √m * mechanism.σ * (√d + √(-2log(α))) / √σmax
    RegParams(ρmin=σmax, ρmax=3σmax, γ=γ, shift=2σmax)
end

#=========================================================================#
# Wishart Mechanism

struct WishartMechanism
    dist :: Distributions.Wishart{Float64, PDMats.ScalMat{Float64}}
    function WishartMechanism(dp::DiffPrivParams; m=1)
        m > 1 && (dp = advanced_composition(m, dp; δfail=dp.δ/(m+1)))
        k = dp.dim + floor(Int, 28log(4/dp.δ)/dp.ε^2)
        new(Distributions.Wishart(k, PDMats.ScalMat(dp.dim, dp.L̃^2)))
    end
end

function regparams(mechanism::WishartMechanism; α, m=1)
    (k, S) = Distributions.params(mechanism.dist)
    d = PDMats.dim(S) - 1
    Δ = √d + √(2(log(4) - log(α)))
    RegParams(
        ρmin = eigmin(S) * (√(m*k) - Δ)^2,
        ρmax = eigmax(S) * (√(m*k) + Δ)^2,
        γ = √eigmax(S) * (√d + √(-2log(α)))
    )
end

(noise::WishartMechanism)() = Symmetric(rand(noise.dist))

#=========================================================================#
# Shifted Mechanisms

struct Shifted{Mechanism}
    mechanism :: Mechanism
    ρmin :: Float64
    Shifted(mechanism; ρmin) = new{typeof(mechanism)}(mechanism, ρmin)
end

Shifted{Mechanism}(dp::DiffPrivParams; m=1, ρmin) where {Mechanism} =
    Shifted(Mechanism(dp; m=m); ρmin=ρmin)

function regparams(shifted::Shifted; α, m=1)
    unshifted = regparams(shifted.mechanism; α=α, m=m)
    shift = shifted.ρmin - unshifted.ρmin
    RegParams(
        ρmin = shifted.ρmin,
        ρmax = unshifted.ρmax + shift,
        γ = unshifted.γ * √(unshifted.ρmin / shifted.ρmin),
        shift = unshifted.shift + shift
    )
end

(shifted::Shifted)() = shifted.mechanism()

function shifted(Mechanism; ρmin)
    function (dp::DiffPrivParams; m=1)
        Shifted{Mechanism}(dp; m=m, ρmin=ρmin)
    end
end

#=========================================================================#
# Shifted Wishart mechanism

struct ShiftedWishart
    mechanism :: WishartMechanism
    ShiftedWishart(dp::DiffPrivParams; m=1) = new(WishartMechanism(dp; m=m))
end

function regparams(shifted::ShiftedWishart; α, m=1)
    (k, S) = Distributions.params(shifted.mechanism.dist)
    d = PDMats.dim(S) - 1
    Δ = √d + √(2(log(4) - log(α)))
    ρmin = 4 * eigmin(S) * √(m*k) * Δ
    regparams(Shifted(shifted.mechanism; ρmin=ρmin); α=α, m=m)
end

(shifted::ShiftedWishart)() = shifted.mechanism()

#=========================================================================#
# Optimizing the shift parameter

struct OptShifted{Mechanism}
    env :: EnvParams
    horizon :: Int
    α :: Float64
end

OptShifted{Mechanism}(env, horizon) where Mechanism =
    OptShifted{Mechanism}(env, horizon, 1/horizon)

function (s::OptShifted{Mechanism})(dp::DiffPrivParams; m=1) where Mechanism
    mechanism = Mechanism(dp; m=m)
    α = s.α/2
    ρmin = regparams(mechanism; α=α/s.horizon, m=m).ρmin
    ρmin_lo = 1.0
    ρmin_hi = max(1.0, ρmin^2)
    opt_regret = optimize(log(ρmin_lo), log(ρmin_hi)) do log_ρmin
        shifted = Shifted(mechanism; ρmin=exp(log_ρmin))
        reg = regparams(shifted; α=α/s.horizon, m=m)
        regret_bound(s.env, reg, s.horizon, α)
    end
    opt_ρmin = exp(minimizer(opt_regret))
    Shifted(mechanism; ρmin=opt_ρmin)
end

#=========================================================================#
# Utility functions

nbits(x) = trailing_zeros(nextpow(2, x+1))

struct IntBits
    bits :: Int
end

Base.eltype(::Type{IntBits}) = Int

Base.length(it::IntBits) = count_ones(it.bits)

function Base.iterate(it::IntBits, s=0)
    bits = it.bits >> s
    if bits == 0
        nothing
    else
        bitpos = 1 + s + trailing_zeros(bits)
        (bitpos, bitpos)
    end
end

#=========================================================================#
# Test functions

noisefunc() = let i = 0; () -> (i -= 1) end

teststrategy() = PanPrivTreeStrategy(8, ReduceStrategy(Int[], push!), noisefunc())

#=========================================================================#

end # module
