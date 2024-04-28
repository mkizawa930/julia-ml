using MLDatasets
using Images
using Flux


traindata = MLDatasets.MNIST(split=:train) # (; features, targets)
X_train, y_train = traindata.features, traindata.targets

y_train = Flux.onehotbatch(y_train, 0:9)

testdata = MNIST(split=:test)
X_test, y_test = testdata.features, testdata.targets

model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu),
    softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

optimizer = Adam(0.0001)

parameters = Flux.params(model)

train_data = [( Flux.flatten(X_train), Flux.flatten(y_train) )]

for i in 1:400
    Flux.train!(loss, parameters, train_data, optimizer)
end

test_data = [(Flux.flatten(X_test), y_test)]