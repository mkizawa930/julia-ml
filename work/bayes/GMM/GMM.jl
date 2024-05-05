module GMM

using Distributions
using CairoMakie
using Random

import Base: rand
import Distributions: pdf
import CairoMakie: plot


"""
Model{D,K}
"""
struct Model{D <: Union{Normal,MvNormal}, K}
    w::Vector{Float64}
    components::Vector{<:D}
end

function Model(w::Vector{T}, μs::Vector{Vector{T}}, Σs::Vector{Matrix{T}}) where T
    @assert length(μs) > 0
    @assert length(Σs) > 0
    @assert length(w) > 0
    @assert length(w) == length(μs) == length(Σs)
    
    K = length(μs[1])
    n_components = length(w)
    ds = [MvNormal(μ, Σ) for (μ, Σ) in zip(μs, Σs)]
    Model{typeof(ds[1]), K}(w, ds)
end


# 混合ガウスモデルのToyモデル
function example()
    π = [0.45, 0.25, 0.3] # 混合率
    # 平均ベクトル
    μs = [[5.0, 35.0], [-20.0, -10.], [30., -20.]] 
    # 共分散行列
    Σs = [[250.0 65.0; 65.0 270.0],
        [125.0 -45.0; -45.0 175.0],
        [125.0 -45.0; -45.0 175.0]] 

    return Model(π, μs, Σs)
end


"""
GaussianMixtureモデルのpdfの値を取得

xs: 入力ベクトル列[(x1,x2), ...]
"""
function pdf(m::Model{D,K}, xs::AbstractVector) where {D,K}
    y = zeros(length(xs))
    for component in m.components
        y .+= pdf.(Ref(component), xs)
    end
    y
end

"""
ランダムサンプリング
"""
function rand(m::Model{D,K}, n::Integer) where {D,K}
    xs = zeros(Float64, K, n)
    cat = Categorical(m.w)
    groups = rand(cat, n)
    for i in 1:length(m.components)
        idxs = findall(x -> x == i, groups)
        d = m.components[i]
        xs[:,idxs] = rand(MvNormal(d.μ, d.Σ), length(idxs))
    end
    xs
end


"""
"""
function plot(model::Model{D,K}) where {D, K}
    # TODO
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

    f = Figure()
    Axis(f[1, 1])
    y = pdf(model, grid)
    contour!(x1_rng, x2_rng, reshape(y, length(x1_rng), length(x2_rng)))

    f
end

end # module