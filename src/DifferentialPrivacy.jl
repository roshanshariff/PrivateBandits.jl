module DifferentialPrivacy

using Distributions: Wishart
using Parameters
using PDMats: ScalMat

using LinearBandits: RegParams
using Accumulators

export ComposedStrategy, TreeStrategy, PanPrivTreeStrategy,
    DiffPrivParams, WishartMechanism, numcomposed,
    advanced_composition, regparams

abstract type ComposedStrategy <: AccumStrategy end

basestrategy(s::ComposedStrategy) = # function stub
    throw(MethodError(basestrategy, typeof((t,))))

numcomposed(T::Type{<:ComposedStrategy}, horizon) = # function stub
    throw(MethodError(numcomposed, Tuple{Type{T}, typeof(horizon)}))

noisetype(T::Type{<:ComposedStrategy}) = # function stub
    throw(MethodError(noisetype, Tuple{Type{T}}))

Accumulators.initial(s::ComposedStrategy) =
    Accumulators.initial(basestrategy(s))

@with_kw struct DiffPrivParams @deftype Float64
    dim::Int
    ε
    δ
    L̃
end

function advanced_composition(m, dp::DiffPrivParams; δfail=p.δ/2)
    @unpack ε, δ = dp
    DiffPrivParams(dp; ε = ε/√(8m*log(1/δfail)), δ = (δ-δfail)/m)
end

function (T::Type{<:ComposedStrategy})(s::AccumStrategy, Noise::Type,
                                       dp::DiffPrivParams, horizon::Int)
    noise = Noise(advanced_composition(numcomposed(T, horizon), dp))
    T(s, noise, horizon)
end

function regparams(T::Type{<:ComposedStrategy}, dp::DiffPrivParams,
                   α, horizon::Int)
    m = numcomposed(T, horizon)
    regparams(noisetype(T), advanced_composition(m, dp), α/horizon; m=m)
end

#=========================================================================#

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
    horizon :: Int
    strategy :: S
    noise :: Noise
    function TreeStrategy(s::AccumStrategy, noise, horizon) where {}
        horizon > 0 || throw(ArgumentError("Horizon $horizon ≤ 0"))
        new{typeof(s), typeof(noise)}(nextpow2(n+1)-1, s, noise)
    end
end

numcomposed(::Type{<:TreeStrategy}, horizon) = nbits(horizon)

noisetype(::Type{TreeStrategy{S, Noise}}) where {S, Noise} = Noise

basestrategy(s::TreeStrategy) = s.strategy

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

struct PanPrivTreeStrategy{S<:AccumStrategy, Noise} <: ComposedStrategy
    horizon :: Int
    strategy :: S
    noise :: Noise
    function PanPrivTreeStrategy(s::AccumStrategy, noise, horizon::Int) where {}
        horizon > 0 || throw(ArgumentError("Horizon $horizon ≤ 0"))
        new{typeof(s), typeof(noise)}(nextpow2(n), s, noise)
    end
end

numcomposed(::Type{<:PanPrivTreeStrategy}, horizon) = 1 + nbits(horizon-1)

noisetype(::Type{PanPrivTreeStrategy{S, Noise}}) where {S, Noise} = Noise

basestrategy(s::PanPrivTreeStrategy) = s.strategy

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

noisefunc() = let i = 0; () -> (i -= 1) end

teststrategy() = PanPrivTreeStrategy(8, ReduceStrategy(Int[], push!), noisefunc())

#=========================================================================#

_wishart_dof(dp::DiffPrivParams) = dp.dim+floor(Int, 28log(4/dp.δ)/dp.ε^2)

struct WishartMechanism
    dist::Wishart{Float64, ScalMat{Float64}}
    WishartMechanism(dp::DiffPrivParams) =
        new(Wishart(_wishart_dof(dp), ScalMat(dp.dim, dp.L̃)))
end

function regparams(::Type{WishartMechanism}, dp::DiffPrivParams, α; m=1)
    L̃² = dp.L̃^2
    k = _wishart_dof(dp)
    d = dp.dim - 1
    Δ = √d + √(2(log(4) - log(α)))
    RegParams(
        ρmin = L̃² * (√(m*k) - Δ)^2,
        ρmax = L̃² * (√(m*k) + Δ)^2,
        γ = dp.L̃ * √m * (√d + √-2log(α))
    )
end

(noise::WishartMechanism)() = Symmetric(rand(noise.dist))

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

end # module
