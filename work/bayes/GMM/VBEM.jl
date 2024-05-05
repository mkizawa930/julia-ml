module VBEM


struct KMeans end
struct EM end
struct GibbsSampling end


# 変分EMアルゴリズム
function fit(x; K, max_iter=10)
    T = eltype(x)
    D, N = size(x)
    params = (
        β0 = 0.01,
        m0 = zeros(T,D),
        ν0 = 1.0,
        α0 = 10*ones(T,K),
        W0 = diagm(ones(T,D)),
        m = [zeros(T,D) for k = 1:K],
        β = 0.001*ones(T,K),
        ν = ones(T,K),
        W = [diagm(ones(T,D)) for k = 1:K],
        α = 10*ones(T,K),
        s = zeros(T,K,N),
        μ = [zeros(T,D) for k = 1:K],
        Λ = [zeros(D,D) for k = 1:K],
        π = [1/K for k = 1:K],
        η = zeros(K,N),       
    )
    # init!(x, params)
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

function init!(x, params)
    @unpack α, s, μ, Λ, π = params
    # init μ
    idxs = sample(1:size(x,2), size(x,1), replace=false)
end


# hyper parameters: m0, β0, ν0, W0, α0
function _vb_em_algorithm!(x, K, params)
    @unpack m0, β0, ν0, W0, α0, m, β, ν, W, η = params # hyper parameter
    @unpack α, s, μ, Λ, π = params
    D, N = size(x)
    
    Λμ = [Λ[k]*m[k] for k = 1:K]
    μTΛμ = [m[k]'*Λ[k]*m[k] for k = 1:K]
    ln_detΛ = [sum(digamma((ν[k]+1-d)/2) for d = 1:D) + D*log(2) + log(det(W[k])) for k = 1:K]
    ln_π = digamma.(α) .- digamma(sum(α))
    @show ln_detΛ, ln_π
    
    # @show Λμ, μTΛμ, ln_detΛ, ln_π
    D, N = size(x)
    # update q(s)=Cat(s|η)
    for n = 1:N
        xn = @view x[:,n]
        for k = 1:K
            η[k,n] = exp(-1/2 * xn'*Λ[k]*xn + xn'*Λμ[k] - 1/2*μTΛμ[k] + 1/2*ln_detΛ[k] + ln_π[k])
        end
        η[:,n] ./= @views sum(η[:,n])
        # calc the expecation with respect to q(s)
        @views s[:,n] = η[:,n]
    end
    
    # q(μ, Λ, π) = q(μ,Λ)q(π)
    # q(μ,Λ) = q(μ|Λ)q(Λ)
    for k = 1:K
        β[k]  = sum(s[k,n] for n = 1:N) + β0
        m[k] .= (sum(s[k,n] * xn for (n,xn) in enumerate(eachcol(x))) + β0*m0) / β[k]
        @show β[k], m[k]
    end
    
    # update q(Λ)
    for k = 1:K
        W_k_inv = sum(s[k,n]*(xn * xn') for (n,xn) in enumerate(eachcol(x)))
        W_k_inv += β0*m0*m0' - β[k]*m[k]*m[k]' + W0^-1
        W[k] .= inv(W_k_inv)
        ν[k] = sum(s[k,n] for n = 1:N) + ν0
        @show ν[k], W_k_inv
    end

    
    # update q(π)
    for k = 1:K
        α[k] = @views sum(s[k,:], dims=2)[1] + α0[k]
    end
    # calc the expectations with respect to  q(μ,Λ)q(π)
    for k = 1:K
        Λ[k] .= ν[k]*W[k]
        display(Λ[k])
        μ[k] .= m[k]
    end
end


end # module