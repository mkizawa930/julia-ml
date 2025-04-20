module VBEM

include("./models.jl")

# 変分EMアルゴリズム
function fit(::VBEMAlgorithm; x, K, max_iter=10)
    T = eltype(x)
    D, N = size(x)
    θ = init_params(D, K, N)

    @unpack m, Λ, ν, W, W0, ν0 = params
    idxs = sample(1:N, K, replace=false)
    m .= [x[:,idx] for idx in idxs]
    Λ .= [ν0*W0 for k = 1:K]
    W .= [copy(W0) for k = 1:K]
    ν .= [ν0 + D for k = 1:K]
    @show params.m
    # init params
    
    for iter in 1:max_iter
        @show iter
        _vb_em_algorithm!(x, K, params)
        draw(x, iter, params)
    end
    params
end

"""
D: 観測変数の次元
K: 混合数
N: 観測数
"""
function init_params(D, K, N)
    return (
        β0 = 0.01,
        m0 = zeros(T,D),
        ν0 = 1.0,
        α0 = 10*ones(T,K),
        W0 = diagm(ones(T,D)),
        m = [zeros(T,D) for _ = 1:K],
        β = 0.001*ones(T,K),
        ν = ones(T,K),
        W = [diagm(ones(T,D)) for _ = 1:K],
        α = 10*ones(T,K),
        s = zeros(T,K,N),
        μ = [zeros(T,D) for _ = 1:K],
        Λ = [zeros(D,D) for _ = 1:K],
        π = [1/K for k = 1:K],
        η = zeros(K,N),  
    )
end

# function init!(x, θ)
#     @unpack α, s, μ, Λ, π = θ
#     # init μ
#     idxs = sample(1:size(x,2), size(x,1), replace=false)
# end

function fit!(; max_iter)
    for iter = 1:max_iter
        estep!()
        mstep!()

        score = calc_elbo()
        # TODO
    end
end




"""
Eステップ: 潜在変数の更新 q(z)
"""
function estep!(xs, θ)
    @unpack m0, β0, ν0, W0, α0, m, β, ν, W, η = params # hyper parameter
    @unpack α, s, μ, Λ, π = params
    D, N = size(x)
    
    Λμ = [Λ[k]*m[k] for k = 1:K]
    μTΛμ = [m[k]'*Λ[k]*m[k] for k = 1:K]
    ln_detΛ = [sum(digamma((ν[k]+1-d)/2) for d = 1:D) + D*log(2) + log(det(W[k])) for k = 1:K]
    ln_π = digamma.(α) .- digamma(sum(α))
    @show ln_detΛ, ln_π
    
    D, N = size(x)
    # q(z): Categorical分布
    for n = 1:N
        xn = @view x[:,n]
        for k = 1:K
            η[k,n] = exp(-1/2 * xn'*Λ[k]*xn + xn'*Λμ[k] - 1/2*μTΛμ[k] + 1/2*ln_detΛ[k] + ln_π[k])
        end
        η[:,n] ./= @views sum(η[:,n])
        # calc the expecation with respect to q(s)
        @views s[:,n] = η[:,n]
    end
end

"""
Mステップ: パラメータの更新 q(μ, Λ, π)
"""
function mstep!(; β,)
   # q(μ, Λ, π) = q(μ,Λ)q(π)
    # q(μ,Λ) = q(μ|Λ)q(Λ) ガウス・ウィシャート分布
    for k = 1:K
        β[k]  = sum(s[k,n] for n = 1:N) + β0
        m[k] .= (sum(s[k,n] * xn for (n,xn) in enumerate(eachcol(x))) + β0*m0) / β[k]
        @show β[k], m[k]
    end
    
    # q(Λ): Wishart分布
    for k = 1:K
        W_k_inv = sum(s[k,n]*(xn * xn') for (n,xn) in enumerate(eachcol(x)))
        W_k_inv += β0*m0*m0' - β[k]*m[k]*m[k]' + W0^-1
        W[k] .= inv(W_k_inv)
        ν[k] = sum(s[k,n] for n = 1:N) + ν0
        @show ν[k], W_k_inv
    end

    # q(μ|Λ): Gaussian分布
    
    # q(π): Dirichlet分布
    for k = 1:K
        α[k] = @views sum(s[k,:], dims=2)[1] + α0[k]
    end
    # calc the expectations with respect to q(μ,Λ)q(π)
    for k = 1:K
        Λ[k] .= ν[k]*W[k]
        display(Λ[k])
        μ[k] .= m[k]
    end
end


end # module