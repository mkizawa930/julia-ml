using LinearAlgebra
using Distributions
using Distributions: Uniform, MvNormal
include("./models.jl")

export fit

struct EMAlgorithm end


"""
EMアルゴリズムによる混合ガウスモデルの推定

Args:
    xs 観測データ
    K 混合数
    seed シード値
"""
function fit(
    ::EMAlgorithm;
    xs::Matrix, 
    n_components, 
    stop_criteria=1e-4, 
    max_iter=30, 
    seed = 123,
) where T
    Random.seed!(seed)

    @assert length(xs) > 0

    K = n_components  # 混合数
    D, N = size(xs) # 観測次元
    q = zeros(K, N)

    # ハイパーパラメータの初期化
    θ = init_params(xs, D, K)

    prev_ln_p = Inf
    for i = 1:max_iter
        @show iter=i
        e_step!(; xs=xs, q=q, N=N, K=K, θ...)
        m_step!(; xs=xs, q=q, N=N, K=K, θ...)

        ln_p = marginal_loglik(; xs=xs, K=K, N=N, θ...)

        # 停止条件
        delta = abs(ln_p - prev_ln_p)
        delta < stop_criteria && break
        prev_ln_p = ln_p
        @show delta
    end

    return GaussianMixture(θ...)
end

"""
対数周辺尤度ln_p(x, θ)を計算する
"""
function marginal_loglik(; xs, K, N, πs, μs, Σs)
    ln_p = 0.0
    models = [MvNormal(μs[k], Σs[k]) for k in 1:K]
    for n = 1:N
        temp = 0.0
        for k = 1:K
            temp += log(πs[k] * pdf(models[k], xs[:,n]))
        end
        ln_p += temp
    end
    ln_p / N
end



"""
ハイパーパラメータの初期化

Args:
    D 次元数
    K 混合数

Returns:
    πs zの事前確率 p(z)
    μs 平均ベクトル
    Σs 共分散行列
    q 事後確率 p(z | x)
"""
function init_params(xs, D, K)
    πs = repeat([1/K], K) # 初期の混合比率
    
    dists = @views [Uniform(minimum(xs[i,:]), maximum(xs[i,:])) for i = 1:D]
    μs = [rand.(dists) for k = 1:K]
    Σs = [diagm(1000.0 * ones(eltype(xs), D)) for k = 1:K]

    (
        πs = πs,
        μs = μs,
        Σs = Σs,
    )
end


"""
Eステップ: 現在のハイパーパラメータで各観測値(x_n)の事後分布p(z | x_n)を計算する
"""
function e_step!(; xs, K, N, q, πs, μs, Σs)
    for n in 1:N
        models = [MvNormal(μs[k], Σs[k]) for k in 1:K]
        for k = 1:K
            q[k,n] = @views πs[k] * pdf(models[k], xs[:,n])
        end
        q[:,n] ./= sum(q[:,n]) # normalize
    end
end

"""
Mステップ: ハイパーパラメータ(μ, Σ, γ)を更新する

Args:
    γ 事後分布
    r 混合比率
    xs 観測データ
    μs 平均ベクトル
"""
function m_step!(; xs, K, N, q, πs, μs, Σs)
    for k in 1:K
        qk_sum = @views sum(qk_n for qk_n in q[k,:])

        μs[k] = @views sum(xs[:,n] * q[k,n] for n in 1:N) / qk_sum
        Σs[k] = @views sum(
            q[k,n] * (xs[:,n] - μs[k])*(xs[:,n] - μs[k])' 
            for n in 1:N
        ) / qk_sum
        Σs[k] = Symmetric(Σs[k])
        πs[k] = qk_sum / N # average
    end
end


