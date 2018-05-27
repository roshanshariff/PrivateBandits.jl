module DifferentialPrivacy

import Distributions
import PDMats

using Parameters

using LinearBandits: RegParams
using Accumulators

export ComposedStrategy, TreeStrategy, PanPrivTreeStrategy,
    DiffPrivParams, WishartMechanism, GaussianMechanism, numcomposed,
    advanced_composition, regparams

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

Accumulators.initial(s::ComposedStrategy) =
    Accumulators.initial(basestrategy(s))

Accumulators.dim(s::ComposedStrategy) = dim(basestrategy(s))

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

function (T::Type{<:ComposedStrategy})(s::AccumStrategy, Noise::Type,
                                       horizon::Int, dp::DiffPrivParams)
    noise = Noise(dp; m=numcomposed(T, horizon))
    T(s, noise, horizon)
end

regparams(s::ComposedStrategy, α; m=1) =
    regparams(noisedist(s), α/horizon(s); m=m*numcomposed(s))

#=========================================================================#
# Tree Strategy (not pan-private)

struct TreeAccum{State}
    count :: Int  # 0 ≤ count < 2^length(state)
    state :: Vector{State}
end

function TreeAccum(s::ComposedStrategy, horizon::Int, init)
    initstate = accumulator(basestrategy(s), init)
    state = Vector{typeof(initstate)}(numcomposed(typeof(s), horizon))
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

Accumulators.accumulator(s::TreeStrategy, init) =
    TreeAccum(s, s.horizon, init)

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
        new{typeof(s), typeof(noise)}(s, noise, nextpow2(horizon))
    end
end

numcomposed(::Type{<:PanPrivTreeStrategy}, horizon) = 1 + nbits(horizon-1)

basestrategy(s::PanPrivTreeStrategy) = s.strategy

noisedist(s::PanPrivTreeStrategy) = s.noise

horizon(s::PanPrivTreeStrategy) = s.horizon

Accumulators.accumulator(s::PanPrivTreeStrategy, init) =
    TreeAccum(s, s.horizon, init)

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
# Wishart Mechanism

struct WishartMechanism
    dist :: Distributions.Wishart{Float64, PDMats.ScalMat{Float64}}
    function WishartMechanism(dp::DiffPrivParams; m=1)
        m > 1 && (dp = advanced_composition(m, dp; δfail=dp.δ/(m+1)))
        new(Distributions.Wishart(dp.dim + floor(Int, 28log(4/dp.δ)/dp.ε^2),
                                  PDMats.ScalMat(dp.dim, dp.L̃^2)))
    end
end

function regparams(mechanism::WishartMechanism, α; m=1)
    (k, S) = Distributions.params(mechanism.dist)
    d = PDMats.dim(S) - 1
    Δ = √d + √(2(log(4) - log(α)))
    RegParams(
        ρmin = eigmin(S) * (√(m*k) - Δ)^2,
        ρmax = eigmax(S) * (√(m*k) + Δ)^2,
        γ = √eigmax(S) * (√d + √-2log(α))
    )
end

(noise::WishartMechanism)() = Symmetric(rand(noise.dist))

#=========================================================================#
# Gaussian mechanism

struct GaussianMechanism
    dim :: Int
    σ :: Float64
    GaussianMechanism(dp::DiffPrivParams; m=1) =
        new(dp.dim, √16m * dp.L̃^2 * log(4/dp.δ) / dp.ε)
end

function (noise::GaussianMechanism)()
    d = noise.dim
    σ = noise.σ
    M = zeros(d, d)
    for i = 1:d
        @inbounds M[i,i] = √2*σ*randn()
    end
    for j = 2:d, i = 1:j-1
        @inbounds M[i,j] = σ*randn()
    end
    Symmetric(M, :U)
end

function regparams(mechanism::GaussianMechanism, α; m=1)
    d = mechanism.dim - 1
    σ_max = mechanism.σ * √2m * (√16d - 2log(α))
    γ = √m * mechanism.σ * (√d + √-2log(α))
    RegParams(
        ρmin = σ_max,
        ρmax = 3σ_max,
        γ = γ/√σ_max,
        shift = 2σ_max
    )
end

#=========================================================================#
# Utility functions

nbits(x) = trailing_zeros(nextpow2(x+1))

struct IntBits
    bits :: Int
end

Base.eltype(::Type{IntBits}) = Int
Base.start(it::IntBits) = 0
Base.done(it::IntBits, s) = it.bits >> s == 0
Base.length(it::IntBits) = count_ones(it.bits)

function Base.next(it::IntBits, s)
    bitpos = 1 + s + trailing_zeros(it.bits >> s)
    (bitpos, bitpos)
end

#=========================================================================#
# Test functions

noisefunc() = let i = 0; () -> (i -= 1) end

teststrategy() = PanPrivTreeStrategy(8, ReduceStrategy(Int[], push!), noisefunc())

#=========================================================================#

end # module
