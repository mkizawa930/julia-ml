module VAE

using Flux
import Flux: params
using Statistics


"""
"""
struct Encoder
    l1::Dense
    l21::Dense
    l22::Dense
end

function Encoder(input_dim, hidden_dim, latent_dim)
    l1 = Dense(input_dim, hidden_dim, relu)
    l21 = Dense(hidden_dim, latent_dim) # for μ
    l22 = Dense(hidden_dim, latent_dim) # for log(σ²)
    Encoder(l1, l21, l22)
end

function (m::Encoder)(x)
    x = Chain(m.l1)(x)
    return m.l21(x), m.l22(x)
end

Flux.@layer Encoder

# function params(m::Encoder)
#     return Flux.params(m.chain, m.l1, m.l2)
# end

struct Decoder
    chain::Chain
end

function Decoder(input_dim, hidden_dim, latent_dim)
    chain = Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, input_dim, sigmoid)
    )
    return Decoder(chain)
end
# params(m::Decoder) = Flux.params(m.chain)

function (m::Decoder)(x)
    return m.chain(x)
end

Flux.@layer Decoder

struct Model
    encoder::Encoder
    decoder::Decoder
end

function Model(input_dim, hidden_dim, latent_dim)
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(input_dim, hidden_dim, latent_dim)
    return Model(encoder, decoder)
end

function (m::Model)(x)
    μ, logσ2 = m.encoder(x)
    z = μ .+ exp.(logσ2 ./ 2) .* randn(size(logσ2))
    x̂ = m.decoder(z)
    return x̂, μ, logσ2
end

Flux.@layer Model

KLD(μ, logσ2) = 1/2 * sum(1 .+ logσ2 .- μ .^ 2 .- exp.(logσ2))

function loss(x, x̂, μ, logσ2; β=1.0)
    reconstruction_error = Flux.binarycrossentropy(x̂, x, agg=sum)
    # reconstruction_error = mean(sum(abs2.(x̂ .- x), dims=2))
    kld = - 1/2 * sum(1 .+ logσ2 .- μ .^ 2 .- exp.(logσ2))
    return reconstruction_error + β * kld
end


end # module