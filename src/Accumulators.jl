module Accumulators

using Base.LinAlg: Cholesky, lowrankupdate!

export AccumStrategy, Accumulator, CholeskyUpdate, SymMatrix,
    accumulator, accum, accum!, accumulated

abstract type AccumStrategy end

accumulator(s::AccumStrategy, init) = # function stub
    throw(MethodError(accumulator, typeof((s, init))))

accum!(s::AccumStrategy, a, x=nothing) = # function stub
    throw(MethodError(accum!, typeof((s, a, x))))

accumulated(s::AccumStrategy, a) = # function stub
    throw(MethodError(accumulated, typeof((s, a))))

#=========================================================================#

accum(s::AccumStrategy, a) = accumulator(s, accumulated(s, a))

accum(s::AccumStrategy, a, x) = accum!(s, accum(s, a), x)

#=========================================================================#

mutable struct Accumulator{Strategy<:AccumStrategy, Accum}
    strategy :: Strategy
    accumulator :: Accum
    function Accumulator(s, init) where {}
        initstate = accumulator(s, init)
        new{typeof(s), typeof(initstate)}(s, initstate)
    end
end

function accum!(a::Accumulator, x)
    a.accumulator = accum!(a.strategy, a.accumulator, x)
    a
end

accumulated(a::Accumulator) = accumulated(a.strategy, a.accumulator)

#=========================================================================#

struct ReduceStrategy{T, Op} <: AccumStrategy
    op! :: Op
end

accumulator(s::ReduceStrategy, init) = copy(init)

accum!(s::ReduceStrategy, a, x) = s.op!(a, x)

accumulated(::ReduceStrategy, a) = a

#=========================================================================#

struct CholeskyUpdate <: AccumStrategy
    dim :: Int
end

accumulator(::CholeskyUpdate, init) = cholfact(init)
accumulator(::CholeskyUpdate, init::Cholesky) = copy(init)
accumulator(s::CholeskyUpdate, init::UniformScaling) =
    Cholesky(diagm(fill(√init.λ, s.dim)), :U)

function accum!(::CholeskyUpdate, a::Cholesky, x::AbstractMatrix)
    M = full(a)
    M .+= x
    cholfact!(M)
end

accum!(::CholeskyUpdate, a::Cholesky, x::StridedVector) =
    lowrankupdate!(a, x)

accum!(::CholeskyUpdate, a::Cholesky, ::Void) = a

accumulated(::CholeskyUpdate, a::Cholesky) = a

#=========================================================================#

struct SymMatrix <: AccumStrategy
    dim :: Int
end

accumulator(::SymMatrix, init) = Symmetric(copy(init))
accumulator(::SymMatrix, init::Symmetric) = copy(init)
accumulator(s::SymMatrix, init::UniformScaling) =
    Symmetric(diagm(fill(init.λ, s.dim)))

function accum!(::SymMatrix, a::Symmetric, x::Symmetric)
    M = parent(a)
    M .+= x
    a
end

function accum!(::SymMatrix, a::Symmetric, x::AbstractVector)
    M = parent(a)
    M .+= x .* x'
    a
end

accum!(::SymMatrix, a::Symmetric, ::Void) = a

accumulated(::SymMatrix, a::Symmetric) = a

end  # module
