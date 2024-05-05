using Distributions
using StatsBase
using Plots
using LinearAlgebra
using UnPack
import StatsBase: loglikelihood
import Distributions: pdf

struct GaussianMixture{T <: Number, K}
    n_components::Int
    w::Vector{Float64}   # mixture
    μ::Vector{Vector{T}} # mean
    Σ::Vector{Matrix{T}} # cov
end
function GaussianMixture(μ::Vector{Vector{T}}, Σ::Vector{Matrix{T}}, w::Vector{Float64}) where T
    @assert length(μ) == length(Σ) == length(w)
    n_components = length(μ)
    GaussianMixture{T,length(μ[1])}(n_components, w, μ, Σ)
end

"""
混合モデルの各混合要素の多変量正規分布オブジェクトのリストを取得する
"""
function get_dists(gm::GaussianMixture{T,K}) where {T,K}
    @unpack μ, Σ = gm
    collect(MvNormal(μk,Σk) for (μk,Σk) in zip(μ,Σ))
end

"""
GaussianMixtureモデルのpdfの値を取得
"""
function pdf(gm::GaussianMixture{T,K}, xs::AbstractVector) where {T,K}
    dist = get_dists(gm)
    y = zeros(T,length(xs))
    for k = 1:K
        y .+= pdf.(Ref(dist[k]), xs)
    end
    y
end

"""
ガウス混合モデルのサンプルを生成する
"""
function toy_problem()
    π = [0.45, 0.25, 0.3] # 混合率
    # 平均ベクトル
    μ = [[5.0, 35.0], [-20.0, -10.], [30., -20.]] 
    # 共分散行列
    Σ = [[250.0 65.0; 65.0 270.0],
        [125.0 -45.0; -45.0 175.0],
        [210.0 -15.0; -15.0 250.0]] 

    return GaussianMixture(μ, Σ, π)
end

"""
    contour_plot(gm::GaussianMixture{T,K}) where {T,K}
2次元ガウス混合モデル(K=2)の等高線をプロットする
"""
function contour_plot(gm::GaussianMixture{T,K}) where {T,K}
    @assert K == 2
    (; μ, Σ) = gm

    # x-y平面の範囲を生成
    x1_rng = begin
        st = minimum(μk[1] for μk in μ) - 3*sqrt(Σ[1][1,1])
        en = maximum(μk[1] for μk in μ) + 3*sqrt(Σ[1][1,1])
        range(st, stop=en, length=300)
    end
    x2_rng = begin
        st = minimum(μk[2] for μk in μ) - 3*sqrt(Σ[2][2,2])
        en = maximum(μk[2] for μk in μ) + 3*sqrt(Σ[2][2,2])
        range(st, stop=en, length=300)
    end
    # グリッドを生成
    grid = collect.(Iterators.product(x1_range, x2_range)) |> vec;
    # 関数値を取得
    f = pdf(gm, grid);
    return contour(x1_rng, x2_rng, f)
end


"""
    em_algorithm()
ガウス混合モデルにおけるEMアルゴリズムによるクラスタリング
"""
function em_algorithm(xs::AbstractVector{T}, K; seed=123) where T
    Random.seed!(seed)
    N = size(xs,2)
    μs, Σs, π = init_params(T, K, length(xs))

end

function init_params(T, K, M)
    # 分散共分散
    Σs = [diagm(1000*ones(eltype(x), dim)) for k = 1:K]
    # 混合率
    π  = repeat([1/K], K)
    # 平均ベクトル
    rngs  = [(minimum(x[i,:]), maximum(x[i,:])) for i = 1:M]
    μs = [rand(Uniform(rng...)) for rng in rngs]
    return μs, Σs, π
end

function _em_algorithm!(xs, μs, Σs, π, K, N, max_iter)
    γ = zeros(Float64, N, K)
    for iter in 1:max_iter
        # e-step
        # update γ
        for (x, n) in enumerate(eachcol(xs))
            w = zeros(K)
            for k = 1:K
                w .= 0.0
                for kk = 1:K
                    loglik = loglikelihood(d, x)
                    w[kk] = π[kk] * exp(loglik)
                end
                γ[n,k] = w[k] / sum(w)
            end
        end

        # m-step
        # update μ
        ratio = zeros(K)
        for k = 1:K
            ratio[k] = sum(γ[n,k] for n = 1:N)
            μs[k] .= 1 / ratio * sum(γ[n,k] * x for (n, x) in enumerate(eachcol(x)))
        end

        # update Σ
        for k = 1:K
            μk = μs[k]
            Σs[k] .= 1 / ratio * sum(γ[n,k] * (x - μk) * (x - μk)' for (n,x) in enumerate(eachcol(x)))
        end

        # udpate π
        for k = 1:K
            π[k] = ratio[k] / N
        end

        # evaluate marginal likelihood
        lnp_new = evaluate!(γ, μs, Σs, π)
        dlnp = lnp_old - lnp_new # 減少した量
        dlnp < threashold && break
    end
end


# サンプルモデルの生成
gm_truth = toy_problem()
# 
contour_plot(gm_truth)


