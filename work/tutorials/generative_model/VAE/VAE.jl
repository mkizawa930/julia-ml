module VAE

using Flux
import Flux: params

input_dim=784
hidden_dim=600
latent_dim=2

struct Encoder
    chain::Chain
    l1::Dense
    l2::Dense
end

function Encoder(input_dim, hidden_dim, latent_dim)
    chain = Chain(
        Dense(input_dim, hidden_dim, softplus)
    )
    l1 = Dense(hidden_dim, latent_dim) # for μ
    l2 = Dense(hidden_dim, latent_dim, exp) # for log(σ²)
    Encoder(chain, l1, l2)
end

function (m::Encoder)(x)
    x = m.chain(x)
    return m.l1(x), m.l2(x)
end

Flux.@layer Encoder

function params(m::Encoder)
    return Flux.params(m.chain, m.l1, m.l2)
end

struct Decoder
    chain::Chain
end

function Decoder(input_dim, hidden_dim, latent_dim)
    chain = Chain(
        Dense(latent_dim, hidden_dim, softplus),
        Dense(hidden_dim, input_dim, sigmoid)
    )
    return Decoder(chain)
end
params(m::Decoder) = Flux.params(m.chain)

function (m::Decoder)(x)
    return m.chain(x)
end

Flux.@layer Decoder

function criterion(x, x̂, μ, logσ2)
    BCE = Flux.crossentropy(x̂, x)
    KLD = - 1/2 * sum(1 + logσ2 - μ .^ 2 - exp.(logσ2))
    return BCE + KLD
end

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
    μ, σ2 = m.encoder(x)

    z = μ .+ sqrt.(σ2) .* randn(size(σ2))
    
    x̂ = m.decoder(z)
    return x̂
end

Flux.@layer Model

end # module