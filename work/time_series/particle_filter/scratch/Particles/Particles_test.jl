using Test

include("./Particles.jl")
using .Particles

ps = Particles.ParticleSet(zeros(4,100), Dict(:x => 1:3, :α => 3:3))

ps[:x]
ps[:α]