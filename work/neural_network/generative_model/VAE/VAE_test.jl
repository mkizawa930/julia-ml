using Test
using Flux
using MLDatasets
using Plots
using Images

train_data = MNIST(split=:train)
test_data = MNIST(split=:test)

X_train, y_train = Flux.flatten(train_data.features), train_data.targets
X_test, y_test = Flux.flatten(test_data.features), test_data.targets

train_dataloader = Flux.DataLoader((X_train, y_train,), batchsize=1000, shuffle=true)
test_dataloader = Flux.DataLoader((X_test, y_test,), batchsize=1000, shuffle=false)

## 学習モデル作成
include("./VAE.jl")

input_dim = 784
hidden_dim = 128
latent_dim = 16
model = VAE.Model(input_dim, hidden_dim, latent_dim)

epochs = 100
opt = Flux.Adam(0.001)
params = Flux.params(model)

for epoch in 1:epochs
    train_loss = 0.0
    val_loss = 0.0

    for (i, data) in enumerate(train_dataloader)
        loss, grads = Flux.withgradient(params) do
            x, _ = data
            x̂, μ, logσ2 = model(x)
            return VAE.loss(x, x̂, μ, logσ2)
        end
        train_loss += loss
        Flux.update!(opt, params, grads)
    end

    train_loss /= length(train_dataloader)
    
    for (x, _) in test_dataloader
        x̂, μ, logσ2 = model(x)
        val_loss += VAE.loss(x, x̂, μ, logσ2)
    end

    val_loss /= length(test_dataloader)

    println("epoch: $epoch, train loss: $train_loss, val loss: $val_loss")
end

## 元画像と生成画像を確認する
x̂ = train_dataloader |> first |> x -> x[1]
reshape(x̂, 28, 28) .|> Gray

