module Particles

using Random
using Distributions
using DataFrames

using StatsBase: quantile
using DataStructures: OrderedDict
using Base: @kwdef

export ParticleFilter, fit

include("./utils.jl")




"""
ParticleFilter{T,U}

Fields:
    n_particles::Int 粒子数
    n_particle_dim::Int 状態次元
    lags::Int 固定ラグ区間
    f::Function 遷移関数 f(x,w)
    h::Function 観測関数 h(x,v)
    init_particle_dists::Dict{Symbol, <:Sampleable} 初期化粒子分布
    sys_noise_dist::T システム雑音分布
    obs_noise_dists::U 観測雑音分布
"""
@kwdef struct ParticleFilter{T<:Distribution, U<:Distribution, }
    n_particles::Int=10000
    n_particle_dim::Int
    n_obs_dim::Int=1
    lags::Int=20
    f::Function
    h::Function
    init_particle_dists::Dict{Symbol, <:Sampleable}
    sys_noise_dist::T
    obs_noise_dist::U
end


struct Results
    sampled::Dict{Symbol,Any}
    data::DataFrame
end



"""
粒子フィルタによるフィルタリング
"""
function fit(pf::ParticleFilter, y::AbstractArray, seed=123)
    Random.seed!(seed)
    if ndims(y) == 1
        y = reshape(y, 1, :)
    end
    N = size(y,2) # sequence length

    # 粒子初期化
    pars = init_particles(pf) # 粒子格納配列
    pars_memory = zeros(size(pars)..., pf.lags) 
    pred_pars = copy(pars)    # 予測粒子格納配列
    weights = zeros(pf.n_particles) # 粒子重み格納配列

    # 観測ノイズ分布モデルの生成
    obs_noise_dist_factory = DistributionFactory(pf.obs_noise_dist, )
    results = (
        particle = Dict{Symbol,Any}(
            :predicted => [],
            :resampled => [],
            :smoothed => [],
        ),
    )
    
    for k = 1:N
        # 遷移
        pars = update(pf, pars)
        pars_memory[:,:,1] .= pars # copy

        # 予測
        pred_pars = observe(pf, pars)
        # push!(results.particle[:predicted], pred_pars)

        # 粒子重み計算
        weights = calc_weights!(weights, y[k], pred_pars, obs_noise_dist_factory)

        # リサンプル
        pars_memory = resample(pars_memory, weights)
        @views pars .= pars_memory[:,:,1] # copy
        push!(results.particle[:resampled], pars)
        
        # 固定ラグ区間の推移
        if k >= pf.lags # max buffer size
            push!(results.particle[:smoothed], pars_memory[:,:,end])
        end
        pars_memory .= circshift_at(pars_memory, 3, 1)
    end
    # 平滑化が足りない部分は最後にリサンプルした結果を利用する
    for i in 1:pf.lags-1
        push!(results.particle[:smoothed], pars_memory[:,:,end-i+1])
    end

    return results
end



"""
粒子の初期化
"""
function init_particles(pf::ParticleFilter{T, U}) where {T, U}
    (; n_particles, n_particle_dim) = pf
    # 初期分布
    dist = pf.init_particle_dists[:state]
    return rand(dist, n_particle_dim, n_particles)
end



"""
粒子の更新(状態遷移)
"""
function update(pf::ParticleFilter, pars::AbstractArray)
    pars = pf.f(pars) # transition
    pars .+= rand(pf.sys_noise_dist, size(pars)...)
    pars
end

"""
予測粒子
"""
function observe(pf::ParticleFilter, pars::AbstractArray)
    return pf.h(pars)
end


"""
calculate weights for particles
"""
function calc_weights!(weights, y, pars, obs_noise_dist_gen::Factory)
    # weights: 重み配列
    # y: 観測値 y_{k}
    # pars: 予測粒子 p_{k | k-1}
    # obs_noise_dist_gen v_{k}
    
    # distは観測値に対する条件付き確率分布p(y|x)を生成する
    n_obs_dim = size(y,1)
    maxval = -Inf
    if n_obs_dim == 1
        for (i, par) in enumerate(pars)
            model = get_instance(obs_noise_dist_gen)
            weights[i] = Distributions.loglikelihood(model, y - par)
            # if maxval < weights[i]
            #     maxval = weights[i]
            # end
        end
    else
        error("error")
        # for (i, par) in enumerate(eachcol(pars))
        #     cdist(par[1])
        #     # weights[i] = loglikelihood()
        # end
    end
    # normalize
    weights .= exp.(weights .- maximum(weights))
    # @show weights
    weights ./= sum(weights)
    return weights
end

"""
calculate the marginal Log-likelihood
"""
function calc_marginal_loglikelihood!(lp::Float64, weights)::Float64
    lp += sum(weights)
    return lp
end

"""
resample particles
"""
function resample(pars::AbstractArray{T,2}, weights) where {T}
    return pars[:, rand(Categorical(weights), size(pars,2))]
end

function resample(pars::AbstractArray{T,3}, weights) where {T}
    return pars[:, rand(Categorical(weights), size(pars,2)),:]
end


"""
配列Aの指定した位置の次元をシフトする
"""
function circshift_at(A::AbstractArray, at::Int, delta::Int)
    @assert at <= ndims(A)
    return circshift(A, ntuple(i -> ifelse(i == at, delta, 0), ndims(A)))
end

"""
squeeze
"""
squeeze(A::AbstractArray) = dropdims(A, dims=tuple(findall(size(A) .== 1)...))





end # module Particles
