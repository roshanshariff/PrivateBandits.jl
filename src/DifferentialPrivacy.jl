module DifferentialPrivacy

using Parameters
import Distributions
import PDMats
using Optim: optimize, minimizer

using LinearBandits: EnvParams, RegParams, regret_bound
using Accumulators

export TreeStrategy, PanPrivTreeStrategy, DiffPrivParams,
    WishartMechanism, ShiftedWishart, ShiftedWishartA,
    ShiftedWishartB, ShiftedWishartOpt, GaussianMechanism,
    ShiftedGaussian, ShiftedGaussianOpt, regparams

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

regparams(s::ComposedStrategy, α; m=1) =
    regparams(noisedist(s), α/horizon(s); m=m*numcomposed(s))

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
    state = Vector{typeof(initstate)}(numcomposed(s))
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
        new{typeof(s), typeof(noise)}(s, noise, nextpow2(horizon))
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
# Wishart Mechanism

struct WishartMechanism
    dist :: Distributions.Wishart{Float64, PDMats.ScalMat{Float64}}
    function WishartMechanism(dp::DiffPrivParams; m=1)
        m > 1 && (dp = advanced_composition(m, dp; δfail=dp.δ/(m+1)))
        k = dp.dim + floor(Int, 28log(4/dp.δ)/dp.ε^2)
        new(Distributions.Wishart(k, PDMats.ScalMat(dp.dim, dp.L̃^2)))
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
# Shifted Wishart Mechanism
# Shift by ρmin - x for
#  x = 4√mkd
#  x = √(ρmin⋅d)

struct ShiftedWishartMechanism
    dist :: WishartMechanism
    ρmin :: Float64
end

function regparams(shifted::ShiftedWishartMechanism, α; m=1)
    unshifted = regparams(shifted.dist, α; m=m)
    RegParams(
        ρmin = shifted.ρmin,
        ρmax = unshifted.ρmax - unshifted.ρmin + shifted.ρmin,
        γ = unshifted.γ * √(unshifted.ρmin / shifted.ρmin),
        shift = shifted.ρmin - unshifted.ρmin
    )
end

(noise::ShiftedWishartMechanism)() = noise.dist()

struct ShiftedWishart
    ρmin :: Float64
end

function (s::ShiftedWishart)(dp::DiffPrivParams; m=1)
    ShiftedWishartMechanism(WishartMechanism(dp; m=m), s.ρmin)
end

function ShiftedWishartA(dp::DiffPrivParams; m=1)
    wishart = WishartMechanism(dp; m=m)
    (k, S) = Distributions.params(wishart.dist)
    ShiftedWishartMechanism(wishart, 4*√(m*k*(dp.dim-1)))
end

function ShiftedWishartB(dp::DiffPrivParams; m=1)
    dist = WishartMechanism(dp; m=m)
    reg = regparams(dist, 1; m=m)
    ShiftedWishartMechanism(dist, (dp.dim-1)*√reg.ρmin)
end

struct ShiftedWishartOpt
    env :: EnvParams
    horizon :: Int
    α :: Float64
    ShiftedWishartOpt(env, horizon, α=1/horizon) = new(env, horizon, α)
end

function (s::ShiftedWishartOpt)(dp::DiffPrivParams; m=1)
    wishart = WishartMechanism(dp; m=m)
    (k,) = Distributions.params(wishart.dist)
    α = s.α/2
    function regret(log_ρmin)
        mechanism = ShiftedWishartMechanism(wishart, exp(log_ρmin))
        reg = regparams(mechanism, α/s.horizon; m=m)
        regret_bound(s.env, reg, s.horizon, α)
    end
    optresult = optimize(regret, log(√(m*k)),
                         log(regparams(wishart, α/s.horizon; m=m).ρmin))
    opt_ρmin = exp(minimizer(optresult))
    ShiftedWishartMechanism(wishart, opt_ρmin)
end

#=========================================================================#
# Gaussian mechanism

struct GaussianMechanism
    dim :: Int
    σ :: Float64
    ρmin_over_σmax :: Float64
    GaussianMechanism(dp::DiffPrivParams, ρmin_over_σmax=1; m=1) =
        new(dp.dim, √16m * dp.L̃^2 * log(4/dp.δ) / dp.ε, ρmin_over_σmax)
end

function (noise::GaussianMechanism)()
    d = noise.dim
    σ = noise.σ
    M = zeros(d, d)
    for j = 1:d
        for i = 1:j-1
            @inbounds M[i,j] = σ*randn()
        end
        @inbounds M[j,j] = √2*σ*randn()
    end
    Symmetric(M, :U)
end

function regparams(mechanism::GaussianMechanism, α; m=1)
    d = mechanism.dim - 1
    σmax = mechanism.σ * √2m * (√16d - 2log(α))
    ρmin = σmax * mechanism.ρmin_over_σmax
    γ = √m * mechanism.σ * (√d + √-2log(α)) / √ρmin
    RegParams(ρmin=ρmin, ρmax=ρmin+2σmax, γ=γ, shift=ρmin+σmax)
end

struct ShiftedGaussian
    ρmin_over_σmax :: Float64
end

function (s::ShiftedGaussian)(dp::DiffPrivParams; m=1)
    GaussianMechanism(dp, s.ρmin_over_σmax; m=m)
end

struct ShiftedGaussianOpt
    env :: EnvParams
    horizon :: Int
    α :: Float64
    ShiftedGaussianOpt(env, horizon, α=1/horizon) = new(env, horizon, α)
end

function (s::ShiftedGaussianOpt)(dp::DiffPrivParams; m=1)
    α = s.α/2
    σmax = regparams(GaussianMechanism(dp; m=m), α/s.horizon; m=m).ρmin
    function regret(log_ρmin)
        mechanism = GaussianMechanism(dp, exp(log_ρmin)/σmax; m=m)
        reg = regparams(mechanism, α/s.horizon; m=m)
        regret_bound(s.env, reg, s.horizon, α)
    end
    optresult = optimize(regret, log(σmax)/2, 3log(σmax)/2)
    opt_ρmin = exp(minimizer(optresult))
    GaussianMechanism(dp, opt_ρmin/σmax; m=m)
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
