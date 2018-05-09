module LinearBandits

using Base.LinAlg: Cholesky, lowrankupdate!
using DifferentialPrivacy
using PDMats: colwise_dot!, colwise_sumsq!
using Distributions: Beta, Truncated

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

mutable struct Bandit{Alg<:ContextLinBandit, State}
    alg :: Alg
    state :: State
    function Bandit(alg::ContextLinBandit) where {}
        state = initialize(alg)
        new{typeof(alg), typeof(state)}(alg, state)
    end
end

function oneround!(b::Bandit, θ, arms, noise=1.0)
    i = choosearm(b.alg, b.state, arms)
    rewards = At_mul_B(arms, θ)
    let arm = view(arms, :, i)
        b.state = learn!(b.alg, b.state, arm, rewards[i] + noise*randn())
    end
    rewards .= maximum(rewards) .- rewards
    #println("Arm regrets = ", armregrets, ", chose arm ", chosenarm, ", got regret ", armregrets[chosenarm])
    rewards[i]
end


#=========================================================================#

struct EllipLinUCB <: ContextLinUCB
    dim :: Int
    ρ :: Float64
    ε :: Float64
    logδ :: Float64
    logdetV₀ :: Float64
    EllipLinUCB(dim, horizon; ρ=0.1, maxnormθ=1.0) =
        new(dim, ρ, √ρ * maxnormθ, -log(horizon), dim*log(ρ))
end

dim(alg::EllipLinUCB) = alg.dim

initialize(alg::EllipLinUCB) =
    Cholesky(diagm([fill(√alg.ρ, alg.dim); 0]), 'U')

learn!(alg::EllipLinUCB, C, x, y) = lowrankupdate!(C, [x; y])

#invmatnormsq(mat, vecs) = squeeze(sum(vecs .* (mat \ vecs), 1), 1)

invmatnormsq(mat, vecs) =
    colwise_dot!(similar(vecs, size(vecs, 2)), vecs, mat\vecs)

function armUCB(alg::EllipLinUCB, C, arms)
    V = principle_submatrix(C)
    θ̂ = cholU_ldiv_B!(V, cholU_lastcol(C))
    sqrtβ = alg.ε + √min(0, -2*alg.logδ + logdet(V) - alg.logdetV₀)
    ucb = At_mul_B(arms, θ̂)
    ucb .+= sqrtβ .* sqrt.(invmatnormsq(V, arms))
end

function armUCB_alt(alg::EllipLinUCB, C, arms)
    V = principle_submatrix(C)
    θ̂ = cholU_ldiv_B!(V, cholU_lastcol(C))
    Vfact = UpperTriangular(V.factors)
    sqrtβ = alg.ε + √min(0, -2*alg.logδ + logdet(V) - alg.logdetV₀)
    result = similar(arms, size(arms, 2))
    temp = similar(arms, size(arms, 1))
    for j = 1:size(arms, 2)
        @simd for i = 1:size(arms, 1)
            @inbounds temp[i] = arms[i, j]
        end
        if C.uplo == 'L'
            A_ldiv_B!(Vfact, temp)
            Ac_ldiv_B!(Vfact, temp)
        else
            Ac_ldiv_B!(Vfact, temp)
            A_ldiv_B!(Vfact, temp)
        end
        meanreward = zero(eltype(arms))
        bonus = zero(eltype(arms))
        @simd for i = 1:size(arms, 1)
            @inbounds arms_ij = arms[i, j]
            @inbounds meanreward += arms_ij * θ̂[i]
            @inbounds bonus += arms_ij * temp[i]
        end
        @inbounds result[j] = meanreward + sqrtβ*√bonus
    end
    result
end

#=========================================================================#

# Rademacher arms

function orthoarms(dim)
    arms = eye(dim)
    () -> arms
end

function unitarms(dim, m)
    arms = zeros(dim, m)
    normsq = zeros(1, m)
    function ()
        randn!(arms)
        colwise_sumsq!(normsq, arms, 1.0)
        arms ./= sqrt.(normsq)
    end
end

function gaparms(dim, m, suboptgap=0.0, optreward=1.0)
    arms = zeros(dim, m)
    heads = view(arms, 1:1, :)
    tails = view(arms, 2:dim, :)
    dist = Truncated(Beta((dim-1)/2, (dim-1)/2),
                     -Inf, (1+optreward-suboptgap)/2)
    function ()
        randn!(tails)
        colwise_sumsq!(heads, tails, 1.0)
        tails ./= sqrt.(heads)
        rand!(dist, heads)
        heads .= 2.*heads .- 1
        heads[rand(1:m)] = optreward
        tails .*= sqrt.(1 .- heads.^2)
        arms
    end
end

function constarms(makearms)
    arms = makearms()
    () -> arms
end

function runexperiment(alg, numrounds, makearms, randθ=true, noise=1.0)
    b = Bandit(alg)
    θ = if randθ
        normalize!(randn(dim(alg)))
    else
        [1; zeros(dim(alg) - 1)]
    end
    regrets = [oneround!(b, θ, makearms(), noise) for _ in 1:numrounds]
    cumsum!(regrets, regrets)
    regrets
end

#=========================================================================#

function principle_submatrix(A::AbstractMatrix, ord=1)
    range = Base.OneTo(LinAlg.checksquare(A) - ord)
    view(A, range, range)
end

principle_submatrix(C::Cholesky, ord=1) =
    Cholesky(principle_submatrix(C.factors, ord), C.uplo)

function cholU_lastcol(C::Cholesky)
    d = size(C.factors, 1)
    C.uplo == 'U' ? C.factors[1:d-1, d] : C.factors[d, 1:d-1]
end

function cholU_ldiv_B!(C::Cholesky, B)
    M = C[:UL]
    istriu(M) ? A_ldiv_B!(M, B) : Ac_ldiv_B!(M, B)
    B
end

function cholL_ldiv_B!(C::Cholesky, B)
    M = C[:UL]
    istril(M) ? A_ldiv_B!(M, B) : Ac_ldiv_B!(M, B)
    B
end

end # module
