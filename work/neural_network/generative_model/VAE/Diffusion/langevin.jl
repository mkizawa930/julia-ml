using Random
using Distributions
using Plots

p(x) = exp(-x^2 / 2)

log_p(x) = log(1 / sqrt(2π)) + (-x^2 / 2)

function langevin_sampling(K)
    x = rand()
    for k = 1:K
        x = x + α * ()
    end
end