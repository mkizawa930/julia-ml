using Distributions: Categorical, Normal, MvNormal
using CairoMakie
using Random

import Base: rand
import Distributions: pdf
import CairoMakie: plot


"""
GaussianMixture{M,D,K}
M: モデル分布
D: 次元
K: 混合数
"""
struct GaussianMixture{M <: Union{Normal,MvNormal}, D, K}
    w::Vector{Float64}
    components::Vector{<:M}
end

function GaussianMixture(w::Vector{T}, μs::Vector{Vector{T}}, Σs::Vector{Matrix{T}}) where T
    @assert length(μs) > 0
    @assert length(Σs) > 0
    @assert length(w) > 0
    @assert length(w) == length(μs) == length(Σs)
    
    D = length(μs[1])
    K = length(w)
    ds = [MvNormal(μ, Σ) for (μ, Σ) in zip(μs, Σs)]
    GaussianMixture{typeof(ds[1]), D, K}(w, ds)
end

"""
ランダムサンプリング
"""
function rand(m::GaussianMixture{M,D,K}, n::Integer) where {M,D,K}
    xs = zeros(Float64, D, n)
    categorical = Categorical(m.w)
    groups = rand(categorical, n)
    for i in 1:length(m.components)
        idxs = findall(x -> x == i, groups)
        d = m.components[i]
        xs[:,idxs] = rand(MvNormal(d.μ, d.Σ), length(idxs))
    end
    xs
end


"""
GaussianMixtureモデルのpdfの値を取得

xs: 入力ベクトル列[(x1,x2), ...]
"""
function pdf(m::GaussianMixture{M,D,K}, xs::AbstractVector) where {M,D,K}
    y = zeros(length(xs))
    for component in m.components
        y .+= pdf.(Ref(component), xs)
    end
    y
end



"""
サンプル用の混合ガウスモデルを生成する
"""
function example_model()
    w = [0.45, 0.25, 0.3] # 混合率
    # 平均ベクトル
    μs = [[5.0, 35.0], [-20.0, -10.], [30., -20.]] 
    # 共分散行列
    Σs = [[250.0 65.0; 65.0 270.0],
        [125.0 -45.0; -45.0 175.0],
        [125.0 -45.0; -45.0 175.0]] 

    return GaussianMixture(w, μs, Σs)
end

"""
混合ガウス分布によるデータセットの作成
"""
function make_dataset(model::GaussianMixture{M,D,K}, N::Integer) where {M,D,K}
    Xs = zeros(D, N)
    ys = rand(Categorical(model.w), N)
    for (i, d) in enumerate(model.components)
        idxs = findall(ys .== i)
        Xs[:, idxs] .= rand(d, length(idxs))
    end
    
    return Xs, ys
end



"""
"""
function contour_plot!(model::GaussianMixture{M,D,K}) where {M,D,K}
    μs = [d.μ for d in model.components]
    Σs = [d.Σ for d in model.components]

    # x-y平面の範囲を生成
    x1_rng = begin
        st = minimum(μ[1] for μ in μs) - 3*sqrt(Σs[1][1,1])
        en = maximum(μ[1] for μ in μs) + 3*sqrt(Σs[1][1,1])
        LinRange(st, en, 300)
    end
    x2_rng = begin
        st = minimum(μ[2] for μ in μs) - 3*sqrt(Σs[2][2,2])
        en = maximum(μ[2] for μ in μs) + 3*sqrt(Σs[2][2,2])
        LinRange(st, en, 300)
    end
    # グリッドを生成
    grid = collect.(Iterators.product(x1_rng, x2_rng)) |> vec;

    # 関数値を取得
    y = pdf(model, grid)
    contour!(x1_rng, x2_rng, reshape(y, length(x1_rng), length(x2_rng)))
    f
end