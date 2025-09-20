module LSM

using LinearAlgebra, Random

export LSMNet, train_readout!, predict, get_spectral_radius, ablation_spectral_radius, ablation_connectivity, reset!, step!, collect_states

mutable struct LSMNet
    input_dim::Int
    reservoir_size::Int
    output_dim::Int
    connectivity::Float64
    spectral_radius::Float64
    leak_rate::Float64
    Win::Matrix{Float64}
    Wres::Matrix{Float64}
    Wout::Matrix{Float64}
    bout::Vector{Float64}
    state::Vector{Float64}
end

function LSMNet(input_dim, reservoir_size, output_dim;
             connectivity=0.1, spectral_radius=0.9, leak_rate=0.3, seed=123)
    rng = MersenneTwister(seed)
    Win = rand(rng, reservoir_size, input_dim) .* 2 .- 1
    Wres = zeros(reservoir_size, reservoir_size)
    mask = rand(rng, reservoir_size, reservoir_size) .< connectivity
    Wres[mask] .= rand(rng, sum(mask)) .* 2 .- 1
    eigs = eigvals(Wres)
    radius = maximum(abs.(eigs))
    if radius > 0
        Wres .*= spectral_radius / radius
    end
    Wout = zeros(output_dim, reservoir_size)
    bout = zeros(output_dim)
    state = zeros(reservoir_size)
    return LSMNet(input_dim, reservoir_size, output_dim, connectivity, spectral_radius, leak_rate, Win, Wres, Wout, bout, state)
end

function reset!(lsm::LSMNet)
    lsm.state .= 0.0
end

function step!(lsm::LSMNet, input::AbstractVector)
    preact = lsm.Win * input .+ lsm.Wres * lsm.state
    lsm.state .= (1 - lsm.leak_rate) .* lsm.state .+ lsm.leak_rate .* tanh.(preact)
    return lsm.state
end

function collect_states(lsm::LSMNet, inputs::AbstractMatrix)
    reset!(lsm)
    T = size(inputs, 2)
    states = zeros(lsm.reservoir_size, T)
    for t in 1:T
        states[:,t] .= step!(lsm, view(inputs,:,t))
    end
    return states
end

function train_readout!(lsm::LSMNet, inputs::AbstractMatrix, targets::AbstractMatrix; reg=1e-6)
    states = collect_states(lsm, inputs)
    states_aug = vcat(states, ones(1, size(states,2)))
    XTX = states_aug * states_aug' + reg * I
    XTY = states_aug * targets'
    Waug = XTX \ XTY
    lsm.Wout .= Waug[1:end-1,:]'
    lsm.bout .= Waug[end,:]'
end

function predict(lsm::LSMNet, inputs::AbstractMatrix)
    states = collect_states(lsm, inputs)
    return lsm.Wout * states .+ lsm.bout
end

function get_spectral_radius(lsm::LSMNet)
    return maximum(abs.(eigvals(lsm.Wres)))
end

function ablation_spectral_radius(lsm::LSMNet, radii::AbstractVector, 
        train_inputs, train_targets, test_inputs, test_targets)
    results = []
    orig_Wres = copy(lsm.Wres)
    for r in radii
        eigs = eigvals(orig_Wres)
        orig_radius = maximum(abs.(eigs))
        Wres_new = copy(orig_Wres)
        if orig_radius > 0
            Wres_new .*= r / orig_radius
        end
        lsm.Wres .= Wres_new
        train_readout!(lsm, train_inputs, train_targets)
        preds = predict(lsm, test_inputs)
        mse = mean((preds .- test_targets).^2)
        push!(results, (r, mse))
    end
    lsm.Wres .= orig_Wres
    return results
end

function ablation_connectivity(lsm::LSMNet, connects::AbstractVector, 
        train_inputs, train_targets, test_inputs, test_targets)
    results = []
    orig_Wres = copy(lsm.Wres)
    for c in connects
        rng = MersenneTwister(123)
        mask = rand(rng, lsm.reservoir_size, lsm.reservoir_size) .< c
        Wres_new = zeros(lsm.reservoir_size, lsm.reservoir_size)
        Wres_new[mask] .= rand(rng, sum(mask)) .* 2 .- 1
        eigs = eigvals(Wres_new)
        radius = maximum(abs.(eigs))
        if radius > 0
            Wres_new .*= lsm.spectral_radius / radius
        end
        lsm.Wres .= Wres_new
        train_readout!(lsm, train_inputs, train_targets)
        preds = predict(lsm, test_inputs)
        mse = mean((preds .- test_targets).^2)
        push!(results, (c, mse))
    end
    lsm.Wres .= orig_Wres
    return results
end

end # module