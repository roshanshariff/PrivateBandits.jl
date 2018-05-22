module LinearBandits

using Base.LinAlg: Cholesky
using PDMats: colwise_dot!, colwise_sumsq!
using NamedTuples
using Parameters

using Accumulators

export ContextLinBandit, EllipLinUCB, EnvParams, RegParams, dim,
initialize, learn!, choosearm

#=========================================================================#

abstract type ContextLinBandit end

dim(alg::ContextLinBandit) = # function stub
    throw(MethodError(dim, typeof((b,))))

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
end

@with_kw struct RegParams @deftype Float64
    ρmin
    ρmax
    γ
    shift = 0
end

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
    V = lufact(principle_submatrix(M))
    θ̂ = A_ldiv_B!(V, M[1:d, d+1])
    (V, θ̂, _β(ℰ, logdet(V)))
end

function (ℰ::Ellipsoid)(C::Cholesky)
    V = principle_submatrix(C)
    θ̂ = cholU_ldiv_B!(V, cholU_lastcol(C))
    (V, θ̂, _β(ℰ, logdet(V)))
end

struct EllipLinUCB{S <: AccumStrategy} <: ContextLinUCB
    dim :: Int
    strategy :: AccumStrategy
    ℰ :: Ellipsoid
    shift :: Float64
    function EllipLinUCB(strategy::AccumStrategy, env::EnvParams,
                         reg::RegParams, α) where {}
        new{typeof(strategy)}(env.dim, strategy, Ellipsoid(env, reg, α),
                              reg.shift)
    end
end

dim(alg::EllipLinUCB) = alg.dim

initialize(alg::EllipLinUCB) =
    accum!(alg.strategy, accumulator(alg.strategy),
           Symmetric(Diagonal(fill(alg.shift, dim(alg)+1))))

learn!(alg::EllipLinUCB, state, x, y) =
    accum!(alg.strategy, state, [x; y])

function armUCB(alg::EllipLinUCB, state, arms)
    (V, θ̂, β) = alg.ℰ(accumulated(alg.strategy, state))
    ucb = At_mul_B(arms, θ̂)
    ucb .+= β .* sqrt.(max.(0, invmatnormsq(V, arms)))
end

# function armUCB_alt(alg::EllipLinUCB, C, arms)
#     V = principle_submatrix(C)
#     θ̂ = cholU_ldiv_B!(V, cholU_lastcol(C))
#     Vfact = UpperTriangular(V.factors)
#     sqrtβ = alg.ε + √min(0, -2*alg.logδ + logdet(V) - alg.logdetV₀)
#     result = similar(arms, size(arms, 2))
#     temp = similar(arms, size(arms, 1))
#     for j = 1:size(arms, 2)
#         @simd for i = 1:size(arms, 1)
#             @inbounds temp[i] = arms[i, j]
#         end
#         if C.uplo == 'L'
#             A_ldiv_B!(Vfact, temp)
#             Ac_ldiv_B!(Vfact, temp)
#         else
#             Ac_ldiv_B!(Vfact, temp)
#             A_ldiv_B!(Vfact, temp)
#         end
#         meanreward = zero(eltype(arms))
#         bonus = zero(eltype(arms))
#         @simd for i = 1:size(arms, 1)
#             @inbounds arms_ij = arms[i, j]
#             @inbounds meanreward += arms_ij * θ̂[i]
#             @inbounds bonus += arms_ij * temp[i]
#         end
#         @inbounds result[j] = meanreward + sqrtβ*√bonus
#     end
#     result
# end

#=========================================================================#

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

#invmatnormsq(mat, vecs) = squeeze(sum(vecs .* (mat \ vecs), 1), 1)

function invmatnormsq(C::Cholesky, X)
    LinvX = cholL_ldiv_B!(C, copy(X))
    colwise_sumsq!(similar(X, size(X, 2)), LinvX, 1)
end

invmatnormsq(V, X) = colwise_dot!(similar(X, size(X, 2)), X, V\X)

#=========================================================================#

end # module
