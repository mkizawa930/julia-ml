using Distributions
using DataStructures

"""
Factory{T}

# Examples 

```julia
factory = Factory(init_args...)
model = create(factory)
```

"""
struct Factory{T}
    ref::Ref{T}
    params::OrderedDict{Symbol, Float64}
end

function Factory(::Type{T}; init_args...) where T
    fields = fieldnames(T)
    types = fieldtypes(T)

    if isempty(init_args)
        params = OrderedDict()
        for (field, type) in zip(fields, types)
            if hasmethod(zero, (type,))
                value = zero(type)
            elseif hasmethod(one, (type,))
                value = one(type)
            else
                throw(error("initialization method is not defined for $(type)"))
            end
            params[field] = value
        end
        obj = T(values(params)...)
    else
        params = OrderedDict(init_args...)
        obj = T(values(init_args)...)
    end

    return Factory{typeof(obj)}(Ref(obj), params)
end

function Factory(::Type{T}, θ=nothing) where T
    fields = fieldnames(T)
    θ₀ = OrderedDict(zip(fields, zeros(length(fields))))
    if !isnothing(θ) && length(θ) != length(fields)
        @warn "not match the size of θ"
    end
    if !isnothing(θ)
        θ = merge(θ₀, θ)
    else
        θ = θ₀
    end
    ref = Ref(T(values(θ)...))
    Factory{T}(ref, θ)
end

"""
パラメータの更新
"""
function update!(factory::Factory{T}; kwargs...) where T
    for (key, val) in zip(keys(kwargs), values(kwargs))
        factory.params[key] = val
    end
    factory.ref[] = T(values(factory.params)...)
    return
end

"""
オブジェクトを生成する
"""
function create(factory::Factory{T}) where T
    return factory.ref[]
end