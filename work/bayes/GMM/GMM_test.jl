using CairoMakie
using Test

include("./GMM.jl")

using .GMM

model = GMM.example()
f = CairoMakie.plot(model)
xs = GMM.rand(model, 500)
scatter!(xs[1,:], xs[2,:])
f