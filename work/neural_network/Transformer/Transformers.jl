module Transformers

using Base.Threads
using Flux
using LinearAlgebra
using LinearAlgebra.BLAS

export PositionalEncoding, Attention, Encoder, Decoder, Transformer

"""
位置エンコーディング
"""
struct PositionalEncoding
    A::Array{Float64,2}
end

function PositionalEncoding(max_seq_len, embed_dim)
    A = zeros(Float64, embed_dim, max_seq_len)
    for i in 1:floor(Int, embed_dim/2)
        for pos in 1:max_seq_len
            A[2*i-1,pos] = sin( (pos-1) / (10000 ^ (2i / embed_dim)) )
            A[2*i,pos] = cos( (pos-1) / (10000 ^ (2i / embed_dim)) )
        end
    end
    A = reshape(A, size(A))
    return PositionalEncoding(A)
end

function (pe::PositionalEncoding)(x)
    x .+ pe.A
end

struct MultiHeadAttention
    Wk::Dense
    Wq::Dense
    Wv::Dense
    input_dim::Int
    output_dim::Int
    mask::Function
end



"""
AttentionのFunctorの定義

(attention::Attention)(key, query, value)

inputs:
    key: 
    query: 
    value: 

returns:

"""
function (layer::MultiHeadAttention)(key, query, value)
    (;Wk, Wq, Wv, input_dim, output_dim) = layer

    k_seq_len = size(key)[2]
    q_seq_len = size(query)[2]
    v_seq_len = size(value)[2]
    batch_size = size(key)[3]

    @assert k_seq_len == v_seq_len

    # head_dim, seq_len, batch_size
    K = Wk(key)
    Q = Wq(query) 
    V = Wv(value)

    # split into each heads
    K = reshape(K, head_dim, n_heads, k_seq_len, batch_size)
    Q = reshape(Q, head_dim, n_heads, q_seq_len, batch_size)
    V = reshape(V, head_dim, n_heads, v_seq_len, batch_size)
    

    K_Q = zeros(Float32, k_seq_len, q_seq_len, n_heads, batch_size)
    Threads.@threads for i in axes(K_Q,3) # n_heads
        for j in axes(K_Q, 4) # batch
            @views K_Q[:,:,i,j] .= gemm('T', 'N', K[:,i,:,j], Q[:,i,:,j])
        end
    end

    # dot product on each sequence
    # TODO: mask
    K_Q = softmax.(K_Q) ./ sqrt(oupput_dim)

    S = zeros(Float32, output_dim, n_heads, v_seq_len, batch_size)
    Threads.@threads for i in axes(S,3) # heads
        for j in axes(S, 4) # batch
            @views S[:,:,i,j] .= gemm('N', 'N', V[:,i,:,j], K_Q[:,:,i,j]) # seq_len, single_head_dim
        end
    end
    reshape(S, n_heads*head_dim, seq_len, batch_size)
end

function scaled_dot_product()
    # TODO
end


struct PositionWiseFeedForward
    chain::Chain
end

Flux.@layer PositionWiseFeedForward

function PositionWiseFeedForward(input_dim, hidden_dim)
    Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, input_dim),
    )
end

struct EncoderBlock
    chain::Chain
end

function EncoderBlock(input_dim, embed_dim, max_seq_len, dropout_rate=0.2)
    dropout = Dropout(dropout_rate)
    chain = Chain(
        MultiHeadAttention(n_heads, embed_dim),
        PositionWiseFeedForward(embed_dim, embed_dim * 4),
        x -> x + dropout(x),
        Flux.LayerNorm(embed_dim),
        x -> x + dropout(x),
        Flux.LayerNorm(embed_dim),
    )
    return EncoderBlock(chain)
end

Flux.@layer EncoderBlock

function (enc::EncoderBlock)(x)
end

struct Encoder
    chain::Chain
end

Flux.@layer Encoder

function Encoder(n_blocks)
    Chain(
        Embedding(input_dim, embed_dim),
        PositionalEncoding(embed_dim),
        [EncoderBlock(input_dim, embeddim, max_seq_len) for i in 1:n_blocks]...,
    )
end




end # module

