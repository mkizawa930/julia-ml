using Test

using Flux
include("./Transformers.jl")

using .Transformers

vocab_size = 10
max_seq_len = 10
embed_dim = 512
batch_size = 5

@testset "" begin
    chain = Chain(
        Embedding(vocab_size, embed_dim),
        Transformers.PositionalEncoding(max_seq_len, embed_dim)
    )

    input = rand(1:10, vocab_size, batch_size)

    output = chain(input)
    println(size(output))
end