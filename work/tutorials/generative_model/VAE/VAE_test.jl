using Test
using Flux
include("./VAE.jl")

using MLDatasets

input_dim = 784
hidden_dim = 600
latent_dim = 2

model = VAE.Model(input_dim, hidden_dim, latent_dim)

model(randn(784,1))

