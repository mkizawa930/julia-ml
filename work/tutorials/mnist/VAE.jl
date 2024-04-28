using Flux
using MLDatasets

train_data = MNIST(split=:train)
test_data = MNIST(split=:test)

n_row, n_col, n_data = size(train_data.features)

input_dim = n_row * n_col
hidden_dim = 200


function Net(input_dim, hidden_dim)
    decoder = Chain(
        Dense(input_dim, hidden_dim, relu)
        Relu(hidden_dim, relu),
    )
end
