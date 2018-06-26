module LinearBandits

using Base.LinAlg: Cholesky
using PDMats: colwise_dot!, colwise_sumsq!
using Parameters

using Accumulators

export ContextLinBandit, EllipLinUCB, EnvParams, RegParams,
    regret_bound, initialize, learn!, choosearm

#=========================================================================#

abstract type ContextLinBandit end

initialize(alg::ContextLinBandit) = # function stub
    throw(MethodError(initalize, typeof((alg,))))

learn!(alg::ContextLinBandit, s, x, y) = # function stub
    throw(MethodError(learn!, typeof((alg, s, x, y))))

choosearm(alg::ContextLinBandit, s, arms) = # function stub
    throw(MethodError(choosearm, typeof((alg, s, arms))))

abstract type ContextLinUCB <: ContextLinBandit end

armUCB(alg::ContextLinUCB, s, arms) = # function stub
    throw(MethodError(armUCB, typeof((alg, s, arms))))

choosearm(alg::ContextLinUCB, s, arms) = indmax(armUCB(alg, s, arms))

#=========================================================================#

@with_kw struct EnvParams @deftype Float64
    dim :: Int
    maxθnorm = 1
    maxactionnorm = 1
    maxrewardmean = maxθnorm * maxactionnorm
    maxreward = Inf
    σ = 1
    @assert dim ≥ 0
    @assert maxθnorm ≥ 0
    @assert maxactionnorm ≥ 0
    @assert maxrewardmean ≥ 0
    @assert maxreward ≥ 0
end

@with_kw struct RegParams @deftype Float64
    ρmin
    ρmax
    γ
    shift = 0
    @assert ρmax ≥ ρmin > 0
    @assert γ ≥ 0
end

function regret_bound(env::EnvParams, reg::RegParams, horizon, α)
    n = horizon
    d = env.dim
    S = env.maxθnorm
    L = env.maxactionnorm
    B = env.maxrewardmean
    σ = env.σ
    ρmax = reg.ρmax
    ρmin = reg.ρmin
    γ = reg.γ

    β = σ*√(2log(2/α) + d*log((ρmax/ρmin) + n*L^2/(d*ρmin))) + S*√ρmax + γ
    regret_mult = B * √(8*n*d*log1p(n*L^2/(d*ρmin)))
    return β * regret_mult
end

#=========================================================================#

struct Ellipsoid
    σ :: Float64
    κ :: Float64
    ξ :: Float64
    function Ellipsoid(env::EnvParams, reg::RegParams, α)
        κ = 2(log(2) - log(α)) - env.dim*log(reg.ρmin)
        ξ = env.maxθnorm * √reg.ρmax + reg.γ
        new(env.σ, κ, ξ)
    end
end

_β(ℰ::Ellipsoid, logdetV) = ℰ.ξ + ℰ.σ*√max(0, logdetV + ℰ.κ)

function (ℰ::Ellipsoid)(M::Symmetric)
    d = LinAlg.checksquare(M) - 1
    V = lufact(principle_submatrix(parent(M)))
    θ̂ = A_ldiv_B!(V, M[1:d, d+1])
    (V, θ̂, _β(ℰ, logdet(V)))
end

function (ℰ::Ellipsoid)(C::Cholesky)
    V = principle_submatrix(C)
    θ̂ = cholU_ldiv_B!(V, cholU_lastcol(C))
    (V, θ̂, _β(ℰ, logdet(V)))
end

struct EllipLinUCB{S <: AccumStrategy} <: ContextLinUCB
    strategy :: S
    ℰ :: Ellipsoid
    shift :: Float64
    function EllipLinUCB(strategy::AccumStrategy, env::EnvParams,
                         reg::RegParams, α) where {}
        new{typeof(strategy)}(strategy, Ellipsoid(env, reg, α), reg.shift)
    end
end

initialize(alg::EllipLinUCB) =
    accum!(alg.strategy, accumulator(alg.strategy, alg.shift*I))

learn!(alg::EllipLinUCB, state, x, y) = accum!(alg.strategy, state, [x; y])

function armUCB(alg::EllipLinUCB, state, arms)
    (V, θ̂, β) = alg.ℰ(accumulated(alg.strategy, state))
    ucb = At_mul_B(arms, θ̂)
    ucb .+= β .* .√max.(0, invquad(V, arms))
end

#=========================================================================#
# Utility functions

function principle_submatrix(A::AbstractMatrix, ord=1)
    range = Base.OneTo(LinAlg.checksquare(A) - ord)
    view(A, range, range)
end

principle_submatrix(C::Cholesky, ord=1) =
    Cholesky(principle_submatrix(C.factors, ord), C.uplo)

principle_submatrix(S::Symmetric, ord=1) =
    Symmetric(principle_submatrix(parent(S), ord), Symbol(S.uplo))

function cholU_lastcol(C::Cholesky)
    d = size(C.factors, 1)
    C.uplo == 'U' ? C.factors[1:d-1, d] : C.factors[d, 1:d-1]
end

function cholU_ldiv_B!(C::Cholesky, B)
    M = C[:UL]
    istriu(M) ? A_ldiv_B!(M, B) : Ac_ldiv_B!(M, B)
end

function cholL_ldiv_B!(C::Cholesky, B)
    M = C[:UL]
    istril(M) ? A_ldiv_B!(M, B) : Ac_ldiv_B!(M, B)
end

#invquad(mat, vecs) = squeeze(sum(vecs .* (mat \ vecs), 1), 1)

function invquad(C::Cholesky, X)
    LinvX = cholL_ldiv_B!(C, copy(X))
    colwise_sumsq!(similar(X, size(X, 2)), LinvX, 1)
end

invquad(V, X) = colwise_dot!(similar(X, size(X, 2)), X, V\X)

#=========================================================================#

end # module
