module Particles

using Base: @kwdef
import Base: getindex

export ParticleFilter, ParticleSet

@kwdef struct ParticleFilter
    particles::Matrix{Float64}
    indices::Dict{Symbol, Int}
    labels::Set{Symbol}
    targets::Dict{Symbol, Any}
    states::Dict{Symbol, Any}
    params::Dict{Symbol, Any}
    f::Function
    h::Function
    self_organized::Bool=false
end

struct ParticleSet
    particles::Matrix{Float64}
    indices::Dict{Symbol, UnitRange{Int}}
end

function getindex(ps::ParticleSet, key::Symbol)
    @view ps.particles[ps.indices[key], :]
end

function get_particles(pf::ParticleFilter, key::Symbol)
    if pf.self_organized
        return pf.params[key]
    end

    index = get_index_by_key(pf, key)
    return @view pf.particles[index, :]
end


function init_particles!(pf::ParticleFilter)
    i = 1
    for state in pf.states
        key, val = state
        idxs = i:i+length(val)-1
        d = priors[key]
        if d isa Distribution
            particles[:,idxs] .= rand(d, n_particles)
        elseif d isa Number || d isa AbstractVector
            particles[:,idxs] .= val
        else
            @error "$typeof(val)"
        end
        i += length(val)
    end
    if self_organized
        for param in pf.params
            key, val = param
            d = init[key]
            idxs = i:i+length(val)-1
            if d isa Distribution
                particles[:,idxs] .= rand(d, n_particles)
            elseif d isa Number || d isa AbstractVector
                particles[:,length(state)+i] .= d
            else
                @error "typeof($v) $v"
            end
            i += length(val)
        end
    end
end

# 粒子フィルタ
function particle_filter(
    y;
    states, 
    params,
    init,
    system_noise,
    f,
    h,
    loglikelihood_func,
    n_particles=10000, 
    self_organized=false
)
    labels = OrderedSet([
        keys(states)..., 
        keys(params)...
    ])
    @assert length(labels) == sum(length(states) + length(params))
    
    n_obs_dim = size(y,2) # 
    n_state_dim = length(states)
    n_param_dim = length(params)

    n_particle_dim = self_organized ? n_state_dim + n_param_dim : n_state_dim # 粒子数
    @info "n_particle_dim=$n_particle_dim"
    particles = zeros(n_particles, n_particle_dim)  # 粒子配列
    predicted = zeros(n_particles, n_obs_dim) # 予測粒子
    weights = zeros(n_particles) # 粒子重み
    normed_weights = zero(weights) # 粒子重み(正規化)

    # 各時刻の粒子の統計量を保存する変数
    results = OrderedDict{Symbol, DataFrame}(l => DataFrame() for l in labels)
    results[:loglik] = DataFrame()

    # 粒子初期化
    init_particles!(particles, states, params, init; self_organized=self_organized)
    
    for i in axes(y,1)
        # 状態遷移
        for j in axes(particles, 2)
            j > 1 && break
            for k in axes(particles, 1)
                # τ = sqrt(exp(particles[k,4])) + 1e-9
                τ = particles[k,4] + 1e-12
                x = particles[k,1]
                β = particles[k,3]
                @assert τ >= 0 "$τ"
                v = rand(system_noise(τ))
                particles[k,1] = f(x, β, v)
            end
        end
        
        # 粒子重み計算
        for k in eachindex(weights)
            x = particles[k,1]
            α = particles[k,2]
            predicted[k] = σ = h(x, α)
            weights[k] = loglikelihood_func(y[i], σ)
        end
        push!(results[:loglik], (loglik = sum(w for w in weights if !isinf(w)),))
        # 粒子重みを正規化
        normed_weights .= exp.(weights .- maximum(weights))
        normed_weights ./= sum(normed_weights)

        # 統計量保存
        add_stats!(results, predicted, :y, 1)
        for (index, symbol) in enumerate([:x, :α, :β, :τ])
            add_stats!(results, particles, symbol, index)
        end
        
        # リサンプリング
        new_indices = rand(Categorical(normed_weights), n_particles)
        particles = particles[new_indices,:]
    end
    results
end

# 各粒子の統計量を算出する
function add_stats!(results, particles, label, index)
    if !hasproperty(results, label)
        results[label] = DataFrame()
    end
    push!(results[label], (
        median=median(particles[:,index]),
        q25=quantile(particles[:,index], 0.25),
        q75=quantile(particles[:,index], 0.75),
    ))
end


end # module