module Accumulators

using Base.LinAlg: Cholesky, lowrankupdate!

export AccumStrategy, Accumulator, CholeskyUpdate, SymMatrix,
    accumulator, accum, accum!, accum, accumulated, dim

abstract type AccumStrategy end

initial(s::AccumStrategy) = # function stub
    throw(MethodError(initial, typeof((s,))))

accumulator(s::AccumStrategy, init=initial(s)) = # function stub
    throw(MethodError(accumulator, typeof((s, init))))

accum!(s::AccumStrategy, a, x) = # function stub
    throw(MethodError(accum!, typeof((s, a, x))))

accumulated(s::AccumStrategy, a) = # function stub
    throw(MethodError(accumulated, typeof((s, a))))

#=========================================================================#

accum_copy(s::AccumStrategy, a) = accumulator(s, accumulated(s, a))

accum(s::AccumStrategy, a, x) = accum!(s, accum_copy(s, a), x)

#=========================================================================#

mutable struct Accumulator{Strategy<:AccumStrategy, Accum}
    strategy :: Strategy
    accumulator :: Accum
    Accumulator(s, initstate=accumulator(s)) where {} =
        new{typeof(s), typeof(initstate)}(s, initstate)
end

function accum!(a::Accumulator, x)
    a.accumulator = accum!(a.strategy, a.accumulator, x)
    a
end

accumulated(a::Accumulator) = accumulated(a.strategy, a.accumulator)

#=========================================================================#

struct ReduceStrategy{T, Op} <: AccumStrategy
    initial :: T
    op! :: Op
end

initial(s::ReduceStrategy) = s.initial

accumulator(s::ReduceStrategy, init) = copy(init)

accum!(s::ReduceStrategy, a, x) = s.op!(a, x)

accumulated(::ReduceStrategy, a) = a

#=========================================================================#

struct CholeskyUpdate <: AccumStrategy
    dim :: Int
end

dim(s::CholeskyUpdate) = s.dim

initial(s::CholeskyUpdate) = Cholesky(zeros(dim(s), dim(s)), 'U')

accumulator(::CholeskyUpdate, init) = copy(init)

function accum!(::CholeskyUpdate, a, x)
    M = full(a)
    M .+= x
    cholfact!(M)
end

accum!(::CholeskyUpdate, a, x::Vector) = lowrankupdate!(a, x)

accumulated(::CholeskyUpdate, a) = a

#=========================================================================#

struct SymMatrix <: AccumStrategy
    dim :: Int
end

dim(s::SymMatrix) = s.dim

initial(s::SymMatrix) = Symmetric(zeros(dim(s), dim(s)))

accumulator(::SymMatrix, init) = copy(init)

function accum!(::SymMatrix, a, x::Symmetric)
    M = parent(a)
    M .+= x
    a
end

function accum!(::SymMatrix, a, x::Vector)
    M = parent(a)
    M .+= x .* x'
    a
end

accumulated(::SymMatrix, a) = a

end  # module
