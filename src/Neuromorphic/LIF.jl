module LIF

using Random
using Statistics
using ChainRulesCore
using Zygote

# =========================
# Public API
# =========================
export AbstractNeuron, LIFNeuron
export SurrogateFunction, AtanSurrogate, FastSigmoidSurrogate, surrogate
export rate_encode, poisson_encode
export ce_count_loss, accuracy_rate, l1_rate
export LinearLIFLayer, FeedForwardSNN, forward_time, forward
export TrainingConfig, ModelConfig, SpikingDataLoader, parameters
export SGDOptimizer, AdamOptimizer, RMSpropOptimizer
export train_epoch!, train_snn!, apply!

# =========================
# Neurons and Surrogates
# =========================

abstract type AbstractNeuron end

struct LIFNeuron{T<:Real}
    β::T
    θ::T
    reset::Symbol  # :subtract | :zero | :none
end

abstract type SurrogateFunction end

struct AtanSurrogate{T<:Real} <: SurrogateFunction
    α::T
end
AtanSurrogate() = AtanSurrogate(2f0)

struct FastSigmoidSurrogate{T<:Real} <: SurrogateFunction
    α::T
    κ::T
end
FastSigmoidSurrogate() = FastSigmoidSurrogate(2f0, 1f0)

# Surrogate activations (elementwise, broadcast-friendly)
@inline function surrogate(u, s::AtanSurrogate)
    y = (2f0/Float32(π)) .* atan.(Float32(π) .* s.α .* u ./ 2f0)
    return @. 0.5f0 * (y + 1f0)
end

@inline surrogate(u, s::FastSigmoidSurrogate) = max.(0f0, 1f0 .- abs.(s.α .* u ./ s.κ))  # triangular window

# LIF step with surrogate spikes; returns (spk_sur, v_new)
function lif_step(V::AbstractArray, I::AbstractArray, n::LIFNeuron, surr::SurrogateFunction)
    @assert size(V) == size(I)
    Vnew = n.β .* V .+ I
    s_sur = surrogate(Vnew .- n.θ, surr)  # differentiable spike proxy in [0,1]
    if n.reset === :subtract
        Vnext = Vnew .- s_sur .* n.θ
    elseif n.reset === :zero
        spike_mask = Vnew .>= n.θ
        Vnext = Vnew .* .!spike_mask
    else
        Vnext = Vnew
    end
    return s_sur, Vnext
end

# =========================
# Encoders
# =========================

function rate_encode(x::AbstractArray{T}, Tsteps::Int; rng=Random.default_rng()) where {T<:Real}
    xclamp = clamp.(x, zero(T), one(T))
    out = Vector{typeof(x)}(undef, Tsteps)
    for t in 1:Tsteps
        out[t] = map(y -> rand(rng) < y ? one(T) : zero(T), xclamp)
    end
    return out
end

poisson_encode(x::AbstractArray, Tsteps::Int; rng=Random.default_rng()) = rate_encode(x, Tsteps; rng=rng)

# =========================
# Functional: losses/metrics
# =========================

@inline function logsumexp(a; dims)
    m = maximum(a; dims=dims)
    return m .+ log.(sum(exp.(a .- m); dims=dims))
end

function ce_count_loss(spk_seq::Vector{<:AbstractArray}, y::AbstractVector{Int})
    counts = reduce(+, spk_seq)             # C×B
    C, B = size(counts)
    logits = counts
    lse = logsumexp(logits; dims=1)         # 1×B
    idx = CartesianIndex[]
    for b in 1:B
        push!(idx, CartesianIndex(y[b], b))
    end
    nll = -sum(logits[idx]) + sum(lse)
    return nll / B
end

function accuracy_rate(spk_seq::Vector{<:AbstractArray}, y::AbstractVector{Int})
    counts = reduce(+, spk_seq)
    C, B = size(counts)
    ŷ = map(b -> findmax(view(counts, :, b))[2], 1:B)
    return sum(ŷ .== y) / B
end

l1_rate(spk_seq; λ=1f-3) = λ * mean(abs, reduce(+, spk_seq))

# =========================
# Layers and Models
# =========================

struct LinearLIFLayer{TW,TB,TN,TS}
    W::TW
    b::TB
    neuron::TN
    surrogate::TS
end

struct FeedForwardSNN
    layers::Vector{LinearLIFLayer}
    T::Int
end

# One-step advance for a single layer
function step_layer(layer::LinearLIFLayer, x::AbstractArray, v::AbstractArray)
    u = layer.W * x .+ layer.b
    spk, vnext = lif_step(v, u, layer.neuron, layer.surrogate)
    return spk, vnext
end

# Unroll through time (device-agnostic)
function forward_time(model::FeedForwardSNN, xseq::Vector{<:AbstractArray})
    @assert length(model.layers) >= 1
    C = size(model.layers[end].W, 1)
    B = size(xseq[1], 2)
    # per-layer membrane states allocated on the same device/type as inputs
    vstates = Vector{Any}(undef, length(model.layers))
    for (i, L) in enumerate(model.layers)
        vstates[i] = similar(xseq[1], size(L.W, 1), B)
        fill!(vstates[i], zero(eltype(vstates[i])))
    end
    spk_seq = Vector{Any}(undef, model.T)   # store same array type as intermediate x_t
    x_t = xseq[1]
    for t in 1:model.T
        x_t = xseq[t]  # D×B
        for (ℓ, layer) in enumerate(model.layers)
            spk, vnext = step_layer(layer, x_t, vstates[ℓ])
            vstates[ℓ] = vnext
            x_t = spk
        end
        spk_seq[t] = x_t                    # C×B, same device/type as inputs
    end
    return spk_seq
end

function forward(model::FeedForwardSNN, x::AbstractArray)
    xseq = [x for _ in 1:model.T]
    return forward_time(model, xseq)
end

# =========================
# Training utilities
# =========================

Base.@kwdef struct TrainingConfig{T<:Real}
    n_epochs::Int = 5
    batch_size::Int = 64
    lr::T = 1f-3
end

Base.@kwdef struct ModelConfig
    reset::Symbol = :subtract
end

struct SpikingDataLoader{TX,TY}
    xs::TX
    ys::TY
    batch_size::Int
end
Base.IteratorSize(::Type{SpikingDataLoader}) = Base.HasLength()
Base.length(dl::SpikingDataLoader) = cld(length(dl.xs), dl.batch_size)
function Base.iterate(dl::SpikingDataLoader, st=1)
    st > length(dl.xs) && return nothing
    en = min(st + dl.batch_size - 1, length(dl.xs))
    xsb = dl.xs[st:en]
    ysb = dl.ys[st:en]
    Tsteps = length(xsb[1])
    D = size(xsb[1][1], 1)
    B = length(xsb)
    xbatch = Vector{typeof(xsb[1][1])}(undef, Tsteps)
    for t in 1:Tsteps
        # hcat preserves array type for GPU arrays that implement similar
        xbatch[t] = hcat([xsb[b][t] for b in 1:B]...)
    end
    ybatch = collect(ysb)
    return ((xbatch, ybatch), en+1)
end

# Optimizers with gradient clipping
abstract type AbstractOptimizer end

mutable struct SGDOptimizer{T<:Real} <: AbstractOptimizer
    lr::T; momentum::T; clip::T
    v::Vector{Any}
end
SGDOptimizer(lr::T=1f-3; momentum::T=0.9f0, clip::T=1f0) where {T<:Real} = SGDOptimizer(lr, momentum, clip, Any[])

mutable struct AdamOptimizer{T<:Real} <: AbstractOptimizer
    lr::T; β1::T; β2::T; ϵ::T; clip::T
    m::Vector{Any}; v::Vector{Any}
    t::Int
end
AdamOptimizer(lr::T=1f-3; β1::T=0.9f0, β2::T=0.999f0, ϵ::T=1f-8, clip::T=1f0) where {T<:Real} =
    AdamOptimizer(lr, β1, β2, ϵ, clip, Any[], Any[], 0)

mutable struct RMSpropOptimizer{T<:Real} <: AbstractOptimizer
    lr::T; α::T; ϵ::T; clip::T
    ms::Vector{Any}
end
RMSpropOptimizer(lr::T=1f-3; α::T=0.99f0, ϵ::T=1f-8, clip::T=1f0) where {T<:Real} =
    RMSpropOptimizer(lr, α, ϵ, clip, Any[])

function parameters(model::FeedForwardSNN)
    Ws = [L.W for L in model.layers]
    bs = [L.b for L in model.layers]
    return Ws, bs
end

function clip!(grads::Vector{<:AbstractArray}, clip::Real)
    n2 = zero(eltype(grads[1]))
    for g in grads
        n2 += sum(abs2, g)
    end
    n = sqrt(n2 + eps(eltype(n2)))
    if n > clip
        s = clip / n
        for g in grads
            g .*= s
        end
    end
end

function apply!(opt::SGDOptimizer, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    isempty(opt.v) && (opt.v = [zero.(p) for p in params])
    clip!(grads, opt.clip)
    for i in eachindex(params)
        opt.v[i] .= opt.momentum .* opt.v[i] .+ grads[i]
        params[i] .-= opt.lr .* opt.v[i]
    end
end

function apply!(opt::AdamOptimizer, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    isempty(opt.m) && (opt.m = [zero.(p) for p in params]; opt.v = [zero.(p) for p in params])
    opt.t += 1
    clip!(grads, opt.clip)
    for i in eachindex(params)
        opt.m[i] .= opt.β1 .* opt.m[i] .+ (1 .- opt.β1) .* grads[i]
        opt.v[i] .= opt.β2 .* opt.v[i] .+ (1 .- opt.β2) .* (grads[i].^2)
        m̂ = opt.m[i] ./ (1 - opt.β1^opt.t)
        v̂ = opt.v[i] ./ (1 - opt.β2^opt.t)
        params[i] .-= opt.lr .* m̂ ./ (sqrt.(v̂) .+ opt.ϵ)
    end
end

function apply!(opt::RMSpropOptimizer, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    isempty(opt.ms) && (opt.ms = [zero.(p) for p in params])
    clip!(grads, opt.clip)
    for i in eachindex(params)
        opt.ms[i] .= opt.α .* opt.ms[i] .+ (1 .- opt.α) .* (grads[i].^2)
        params[i] .-= opt.lr .* grads[i] ./ (sqrt.(opt.ms[i]) .+ opt.ϵ)
    end
end

# Loss for one batch (rate-coded classification)
function batch_loss(model::FeedForwardSNN, xbatch::Vector{<:AbstractArray}, ybatch::Vector{Int})
    spk_seq = forward_time(model, xbatch)  # [T][C×B]
    loss = ce_count_loss(spk_seq, ybatch)
    return loss, spk_seq
end

# Training one epoch
function train_epoch!(model::FeedForwardSNN, dl::SpikingDataLoader, opt::AbstractOptimizer)
    Ws, bs = parameters(model)
    pars = vcat(Ws, bs)
    tot_loss, tot_acc = 0f0, 0f0
    nb = length(dl)
    for (xbatch, ybatch) in dl
        loss_fun() = batch_loss(model, xbatch, ybatch)[1]
        # compute grads w.r.t. each parameter array
        grads_tuple = Zygote.gradient(() -> loss_fun(), pars...)
        grads = Any[grads_tuple[i] for i in 1:length(pars)]
        # replace any missing gradients with zeros of same shape
        for i in eachindex(grads)
            if grads[i] === nothing
                grads[i] = zero.(pars[i])
            end
        end
        apply!(opt, pars, grads)
        loss_val, spk = batch_loss(model, xbatch, ybatch)
        acc = accuracy_rate(spk, ybatch)
        tot_loss += loss_val
        tot_acc += acc
    end
    return tot_loss/nb, tot_acc/nb
end

# High-level trainer
function train_snn!(model::FeedForwardSNN, dl::SpikingDataLoader, cfg::TrainingConfig; opt=AdamOptimizer(cfg.lr))
    hist = (loss=Float32[], acc=Float32[])
    for epoch in 1:cfg.n_epochs
        l, a = train_epoch!(model, dl, opt)
        push!(hist.loss, Float32(l))
        push!(hist.acc, Float32(a))
    end
    return hist
end

end # module
