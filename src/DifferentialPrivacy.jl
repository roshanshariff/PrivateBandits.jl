module DifferentialPrivacy

using Base.LinAlg: Cholesky, lowrankupdate!
using Distributions

abstract type AccumStrategy end

accumtype(t::Type{<:AccumStrategy}) = # function stub
    throw(MethodError(accumtype, Tuple{Type{t}}))

initial(s::AccumStrategy) = # function stub
    throw(MethodError(initial, typeof((s,))))

accumulator(s::AccumStrategy, init=initial(s)) = # function stub
    throw(MethodError(accumulator, typeof((s, init))))

accumulate!(s::AccumStrategy, a, x) = # function stub
    throw(MethodError(accumulate!, typeof((s, a, x))))

accumulated(s::AccumStrategy, a) = # function stub
    throw(MethodError(accumulated, typeof((s, a))))

abstract type ComposedStrategy <: AccumStrategy end

basestrategy(s::ComposedStrategy) = # function stub
    throw(MethodError(basestrategy, typeof((t,))))

initial(s::ComposedStrategy) = initial(basestrategy(s))

#=========================================================================#
# Utility functions

accumtype(s::AccumStrategy) = accumtype(typeof(s))

accum_copy(s::AccumStrategy, a) = accumulator(s, accumulated(s, a))

accumulate(s::AccumStrategy, a, x) = accumulate!(s, accum_copy(s, a), x)

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

mutable struct Accumulator{Strategy<:AccumStrategy, Accum}
    strategy :: Strategy
    accumulator :: Accum
    Accumulator(s::AccumStrategy, init=initial(s)) where {} =
        new{typeof(s), accumtype(s)}(s, accumulator(s, init))
end

function accumulate!(a::Accumulator, x)
    a.accumulator = accumulate!(a.strategy, a.accumulator, x)
    a
end

accumulated(a::Accumulator) = accumulated(a.strategy, a.accumulator)

#=========================================================================#

struct TreeStrategy{S<:AccumStrategy, Noise} <: ComposedStrategy
    horizon :: Int
    strategy :: S
    noise :: Noise
    function TreeStrategy(n::Int, s::AccumStrategy, noise) where {}
        n > 0 || throw(ArgumentError("Horizon must be positive"))
        new{typeof(s), typeof(noise)}(nextpow2(n+1)-1, s, noise)
    end
end

struct TreeAccum{State}
    count :: Int  # 0 < count < 2^length(state)
    state :: Vector{State}
end

numcomposed(::Type{TreeStrategy}, horizon) = nbits(horizon)

accumtype(::Type{TreeStrategy{S, N}}) where {S<:AccumStrategy, N} =
    TreeAccum{accumtype(S)}

basestrategy(s::TreeStrategy) = s.strategy

function accumulator(s::TreeStrategy, init)
    numstates = numcomposed(TreeStrategy, s.horizon)
    state = Vector{accumtype(s.strategy)}(numstates)
    state[end] = accumulator(s.strategy, init)
    TreeAccum(0, state)
end

function accumulate!(s::TreeStrategy, a::TreeAccum, x)

    horizon = 1<<length(a.state) - 1
    active = (a.count & (horizon>>1)) ⊻ horizon
    initstate = a.state[1 + trailing_zeros(a.count | ~(horizon>>1))]

    for i in 1:trailing_ones(active)
        a.state[i] = accumulate(s.strategy, initstate, s.noise())
    end

    for i in IntBits(active)
        a.state[i] = accumulate!(s.strategy, a.state[i], x)
    end

    TreeAccum((a.count + 1) & horizon, a.state)
end

function accumulated(s::TreeStrategy, a::TreeAccum)
    horizon = 1<<length(a.state) - 1
    state = a.state[1 + trailing_zeros(a.count | ~(horizon>>1))]
    accumulated(s.strategy, state)
end

#=========================================================================#

struct PanPrivTreeStrategy{S<:AccumStrategy, Noise} <: ComposedStrategy
    horizon :: Int
    strategy :: S
    noise :: Noise
    function PanPrivTreeStrategy(n::Int, s::AccumStrategy, noise) where {}
        n > 0 || throw(ArgumentError("Horizon must be positive"))
        new{typeof(s), typeof(noise)}(nextpow2(n), s, noise)
    end
end

numcomposed(::Type{PanPrivTreeStrategy}, horizon) = 1 + nbits(horizon-1)

accumtype(::Type{PanPrivTreeStrategy{S, N}}) where {S<:AccumStrategy, N} =
    TreeAccum{accumtype(S)}

basestrategy(s::PanPrivTreeStrategy) = s.strategy

function accumulator(s::PanPrivTreeStrategy, init)
    numstates = numcomposed(PanPrivTreeStrategy, s.horizon)
    state = Vector{accumtype(s.strategy)}(numstates)
    state[end] = accumulator(s.strategy, init)
    TreeAccum(0, state)
end

function accumulate!(s::PanPrivTreeStrategy, a::TreeAccum, x)

    mask = 1<<(length(a.state) - 1) - 1
    leafpos = 1 + count_ones(a.count)
    newaccum = leafpos + trailing_zeros(a.count | ~mask)

    for i = length(a.state):-1:newaccum
        a.state[i] = accumulate!(s.strategy, a.state[i], x)
    end

    a.state[newaccum] = accumulate!(s.strategy, a.state[newaccum], s.noise())

    for i = newaccum-1:-1:leafpos
        a.state[i] = accumulate(s.strategy, a.state[i+1], s.noise())
    end

    TreeAccum((a.count + 1) & mask, a.state)
end

function accumulated(s::PanPrivTreeStrategy, a::TreeAccum)
    mask = 1<<(length(a.state) - 1) - 1
    prevcount = (a.count - 1) & mask
    leafpos = 1 + count_ones(prevcount)
    accumulated(s.strategy, a.state[leafpos])
end

#=========================================================================#

struct ReduceStrategy{T, Op} <: AccumStrategy
    initial :: T
    op! :: Op
end

accumtype(::Type{ReduceStrategy{T, Op}}) where {T, Op} = T

initial(s::ReduceStrategy) = s.initial

accumulator(s::ReduceStrategy, init) = copy(init)

accumulate!(s::ReduceStrategy, a, x) = s.op!(a, x)

accumulated(::ReduceStrategy, a) = a

#=========================================================================#

noisefunc() = let i = 0; () -> (i -= 1) end

teststrategy() = PanPrivTreeStrategy(8, ReduceStrategy(Int[], push!), noisefunc())

#=========================================================================#

struct CholeskyUpdate <: DifferentialPrivacy.AccumStrategy
    initial :: Cholesky{Float64, Matrix{Float64}}
end

accumtype(::Type{CholeskyUpdate}) = Cholesky{Float64, Matrix{Float64}}

initial(s::CholeskyUpdate) = s.initial

accumulator(::CholeskyUpdate, init) = copy(init)

accumulate!(::CholeskyUpdate, a, x) = cholfact!(full(a) + x)

accumulate!(::CholeskyUpdate, a, x::Vector) = lowrankupdate!(a, x)

accumulated(::CholeskyUpdate, a) = a

#=========================================================================#

wishartnoise_dof(d, ε, δ) = floor(Int, d + 28log(4/δ)/ε^2)

function composed_privacy(k, ε, δ)
    δ′ = δ/2
    (ε/√(8k*log(1/δ′)), (δ-δ′)/k)
end

#=========================================================================#

end # module
