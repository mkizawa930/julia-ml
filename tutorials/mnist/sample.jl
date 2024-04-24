using MLDatasets
using Images

train_X, train_y = MLDatasets.MNIST.traindata(Float32)

Images.load()

RGB.(train_X[:,:,1])