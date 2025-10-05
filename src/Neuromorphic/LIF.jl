"""

    LIF.jl

Complete Leaky Integrate-and-Fire (LIF) framework with:
- LIF neuron models and exact integration
- Flexible network architectures with custom connectivity
- Multiple training methods (surrogate gradient, STDP, reservoir computing)
- Liquid State Machines with spectral radius control
- Comprehensive ablation study framework
"""

using LinearAlgebra
using Statistics
using Random

# ============================================================================
#  CORE LIF NEURON MODEL
# ============================================================================

"""
    LIFNeuron

Leaky Integrate-and-Fire neuron parameters.

# Governing Equation
τ dV/dt = -(V - E_L) + R*I(t)

Exact integration: V(t+dt) = E_L + (V(t) - E_L)*exp(-dt/τ) + R*I*(1 - exp(-dt/τ))
Spike when V ≥ V_th, reset to V_reset, refractory period t_ref
"""
struct LIFNeuron
    τ::Float64          # Membrane time constant (ms)
    E_L::Float64        # Resting potential (mV)
    R::Float64          # Membrane resistance (MΩ)
    V_th::Float64       # Spike threshold (mV)
    V_reset::Float64    # Reset potential (mV)
    t_ref::Float64      # Refractory period (ms)
    dt::Float64         # Time step (ms)
end

LIFNeuron(; τ=20.0, E_L=-70.0, R=10.0, V_th=-50.0, V_reset=-70.0, t_ref=2.0, dt=0.1) =
    LIFNeuron(τ, E_L, R, V_th, V_reset, t_ref, dt)

"""
    lif_step(V, I, neuron::LIFNeuron)

Exact integration step for LIF dynamics.
"""
function lif_step(V, I, neuron::LIFNeuron)
    α = exp(-neuron.dt / neuron.τ)
    V_new = neuron.E_L + (V - neuron.E_L) * α + neuron.R * I * (1 - α)
    
    spike = V_new ≥ neuron.V_th
    if spike
        V_new = neuron.V_reset
    end
    
    return V_new, spike
end


# ============================================================================
#  LIF NETWORK WITH CUSTOM CONNECTIVITY
# ============================================================================

"""
    LIFNetwork

Spiking neural network of LIF neurons with flexible architecture.

Connectivity is encoded by W matrix (N×N):
- W[i,j] ≠ 0: connection from neuron j to neuron i
- W[i,j] = 0: no connection
"""
mutable struct LIFNetwork
    neuron::LIFNeuron
    N::Int
    W::Matrix{Float64}                      # Recurrent weights (connectivity matrix)
    W_in::Matrix{Float64}                   # Input weights
    W_out::Matrix{Float64}                  # Output weights
    V::Vector{Float64}                      # Membrane potentials
    ref_count::Vector{Float64}              # Refractory counters
    spike_history::Vector{Vector{Float64}}  # Spike times
end

"""
    create_connectivity(N::Int, connectivity::Float64, spectral_radius::Float64;
                       pattern="random", seed=nothing)

Create weight matrix with specified topology and spectral radius.

Patterns: "random", "small_world", "scale_free", "ring", "clustered", "feedforward"
"""
function create_connectivity(N::Int, connectivity::Float64, spectral_radius::Float64;
                            pattern="random", seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    W = zeros(N, N)
    
    if pattern == "random"
        mask = rand(N, N) .< connectivity
        W = mask .* randn(N, N)
    elseif pattern == "feedforward"
        layer_size = N ÷ 3
        W[1:layer_size, layer_size+1:2*layer_size] .= randn(layer_size, layer_size)
        W[layer_size+1:2*layer_size, 2*layer_size+1:end] .= randn(layer_size, N-2*layer_size)
    elseif pattern == "small_world"
        k = max(2, Int(round(N * connectivity)))
        for i in 1:N
            for j in 1:k÷2
                W[i, mod1(i+j, N)] = randn()
                W[i, mod1(i-j, N)] = randn()
            end
        end
        for i in 1:N, j in 1:N
            if W[i,j] != 0 && rand() < 0.1
                new_j = rand(1:N)
                if new_j != i
                    W[i, new_j] = W[i, j]
                    W[i, j] = 0.0
                end
            end
        end
    elseif pattern == "ring"
        for i in 1:N
            W[i, mod1(i+1, N)] = randn()
        end
    elseif pattern == "clustered"
        n_clusters = 5
        cluster_size = N ÷ n_clusters
        for c in 1:n_clusters
            s, e = (c-1)*cluster_size+1, min(c*cluster_size, N)
            for i in s:e, j in s:e
                if i != j && rand() < 0.3
                    W[i, j] = randn()
                end
            end
            if c < n_clusters
                ns, ne = e+1, min((c+1)*cluster_size, N)
                for i in s:e, j in ns:ne
                    if rand() < connectivity/2
                        W[i, j] = randn()
                    end
                end
            end
        end
    else
        error("Unknown pattern: $pattern")
    end
    
    # Remove self-connections
    for i in 1:N
        W[i, i] = 0.0
    end
    
    # Scale to spectral radius
    if spectral_radius > 0
        eigenvalues = eigvals(W)
        current_radius = maximum(abs.(eigenvalues))
        if current_radius > 1e-10
            W = W .* (spectral_radius / current_radius)
        end
    end
    
    return W
end

"""
    LIFNetwork(neuron::LIFNeuron, N::Int, N_input::Int, N_output::Int;
               connectivity="random", p_conn=0.1, spectral_radius=0.0, 
               w_scale=1.0, seed=nothing)

Construct LIF network with custom architecture.

# Arguments
- `connectivity::String`: Pattern ("random", "feedforward", "ring", "small_world", "clustered")
- `p_conn::Float64`: Connection probability
- `spectral_radius::Float64`: Target spectral radius (0 = no scaling)
- `w_scale::Float64`: Weight initialization scale
"""
function LIFNetwork(neuron::LIFNeuron, N::Int, N_input::Int, N_output::Int;
                   connectivity="random", p_conn=0.1, spectral_radius=0.0, 
                   w_scale=1.0, seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    W = create_connectivity(N, p_conn, spectral_radius; pattern=connectivity, seed=seed) .* w_scale
    W_in = randn(N, N_input) .* w_scale
    W_out = randn(N_output, N) .* (0.1 * w_scale)
    
    V = fill(neuron.E_L, N)
    ref_count = zeros(N)
    spike_history = [Float64[] for _ in 1:N]
    
    return LIFNetwork(neuron, N, W, W_in, W_out, V, ref_count, spike_history)
end

function reset!(net::LIFNetwork)
    net.V .= net.neuron.E_L
    net.ref_count .= 0.0
    for i in 1:net.N
        empty!(net.spike_history[i])
    end
end

function step!(net::LIFNetwork, I_ext::Vector{Float64}, t::Float64)
    spikes = falses(net.N)
    I_syn = net.W_in * I_ext
    
    for i in 1:net.N
        if net.ref_count[i] > 0
            net.ref_count[i] -= net.neuron.dt
            continue
        end
        
        for j in 1:net.N
            if length(net.spike_history[j]) > 0 && t - net.spike_history[j][end] < 5.0
                I_syn[i] += net.W[i, j]
            end
        end
        
        net.V[i], spikes[i] = lif_step(net.V[i], I_syn[i], net.neuron)
        
        if spikes[i]
            push!(net.spike_history[i], t)
            net.ref_count[i] = net.neuron.t_ref
        end
    end
    
    return spikes
end

"""
    simulate(net::LIFNetwork, I_input::Matrix{Float64}, T::Float64)

Simulate network over time series.

Returns: (V_trace, spike_trains)
"""
function simulate(net::LIFNetwork, I_input::Matrix{Float64}, T::Float64)
    n_steps = size(I_input, 2)
    V_trace = zeros(net.N, n_steps)
    spike_trains = falses(net.N, n_steps)
    
    reset!(net)
    t = 0.0
    
    for step in 1:n_steps
        spikes = step!(net, I_input[:, step], t)
        V_trace[:, step] .= net.V
        spike_trains[:, step] .= spikes
        t += net.neuron.dt
    end
    
    return V_trace, spike_trains
end


# ============================================================================
#  TRAINING METHODS
# ============================================================================

"""
    train_surrogate_gradient!(net::LIFNetwork, X_train, y_train;
                              epochs=50, η=0.01, surrogate="sigmoid", β=10.0)

Train using surrogate gradient descent (BPTT).

Surrogate types: "sigmoid", "linear", "exponential"
"""
function train_surrogate_gradient!(net::LIFNetwork, X_train, y_train;
                                  epochs=50, η=0.01, surrogate="sigmoid", β=10.0)
    
    println("="^60)
    println("Training with Surrogate Gradient Descent")
    println("Epochs: $epochs, η: $η, Surrogate: $surrogate")
    println("="^60)
    
    function surrogate_derivative(V, V_th, type, β)
        if type == "sigmoid"
            x = β * abs(V - V_th)
            return β * exp(-x) / (1 + exp(-x))^2
        elseif type == "linear"
            return max(0.0, 1.0 - β * abs(V - V_th))
        elseif type == "exponential"
            return β * exp(-β * abs(V - V_th))
        else
            error("Unknown surrogate: $type")
        end
    end
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        for (X, y) in zip(X_train, y_train)
            V_trace, spike_trains = simulate(net, X, size(X, 2) * net.neuron.dt)
            spike_counts = sum(spike_trains, dims=2)[:, 1]
            output = net.W_out * spike_counts
            loss = sum((output .- y).^2) / length(y)
            total_loss += loss
            
            ∂L_∂output = 2 .* (output .- y) ./ length(y)
            ∂L_∂counts = net.W_out' * ∂L_∂output
            
            net.W_out .-= η .* (∂L_∂output * spike_counts')
            
            for i in 1:net.N, j in 1:net.N
                if net.W[i, j] != 0.0
                    grad = 0.0
                    for t in 1:size(V_trace, 2)
                        s_deriv = surrogate_derivative(V_trace[i, t], net.neuron.V_th, surrogate, β)
                        pre_spike = (t > 5 && spike_trains[j, t-5]) ? 1.0 : 0.0
                        grad += ∂L_∂counts[i] * s_deriv * pre_spike
                    end
                    net.W[i, j] -= η * grad / size(V_trace, 2)
                end
            end
        end
        
        if epoch % 10 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(total_loss/length(X_train), digits=6))")
        end
    end
    println("Training complete!\n")
end

"""
    train_stdp!(net::LIFNetwork, X_train; T_train=1000.0, 
                A_plus=0.01, A_minus=0.01, τ_stdp=20.0)

Train using Spike-Timing Dependent Plasticity (unsupervised).

Weight update: Δw = A_plus*exp(-Δt/τ) if pre→post, -A_minus*exp(Δt/τ) if post→pre
"""
function train_stdp!(net::LIFNetwork, X_train; T_train=1000.0,
                    A_plus=0.01, A_minus=0.01, τ_stdp=20.0, w_max=5.0, w_min=-5.0)
    
    println("="^60)
    println("Training with STDP (Unsupervised)")
    println("A_plus: $A_plus, A_minus: $A_minus, τ: $τ_stdp")
    println("="^60)
    
    for (idx, X) in enumerate(X_train)
        simulate(net, X, T_train)
        
        weight_changes = 0
        for i in 1:net.N, j in 1:net.N
            if i == j || net.W[i, j] == 0.0
                continue
            end
            
            Δw = 0.0
            for t_post in net.spike_history[i], t_pre in net.spike_history[j]
                Δt = t_pre - t_post
                if Δt < 0
                    Δw += A_plus * exp(Δt / τ_stdp)
                elseif Δt > 0
                    Δw -= A_minus * exp(-Δt / τ_stdp)
                end
            end
            
            if abs(Δw) > 1e-10
                net.W[i, j] = clamp(net.W[i, j] + Δw, w_min, w_max)
                weight_changes += 1
            end
        end
        
        if idx % 5 == 0 || idx == 1
            println("Sample $idx/$(length(X_train)) - Updates: $weight_changes")
        end
    end
    println("STDP training complete!\n")
end

"""
    train_readout!(net::LIFNetwork, X_train, y_train; λ=0.01, feature="spike_count")

Train output layer via ridge regression (Reservoir Computing / LSM).

Feature types: "spike_count", "final_voltage", "mean_voltage"
"""
function train_readout!(net::LIFNetwork, X_train, y_train; λ=0.01, feature="spike_count")
    println("="^60)
    println("Training Readout (Reservoir Computing)")
    println("Feature: $feature, λ: $λ")
    println("="^60)
    
    n_samples = length(X_train)
    features = zeros(net.N, n_samples)
    targets = zeros(length(y_train[1]), n_samples)
    
    for (idx, (X, y)) in enumerate(zip(X_train, y_train))
        V_trace, spike_trains = simulate(net, X, size(X, 2) * net.neuron.dt)
        
        if feature == "spike_count"
            features[:, idx] .= sum(spike_trains, dims=2)[:, 1]
        elseif feature == "final_voltage"
            features[:, idx] .= V_trace[:, end]
        elseif feature == "mean_voltage"
            features[:, idx] .= mean(V_trace, dims=2)[:, 1]
        end
        
        targets[:, idx] .= y
        
        if idx % 20 == 0 || idx == 1
            println("Processed $idx/$n_samples")
        end
    end
    
    # Ridge regression: W_out = Y * X^T * (X * X^T + λI)^{-1}
    gram = features * features' + λ * I
    net.W_out .= targets * features' * inv(gram)
    
    predictions = net.W_out * features
    train_error = sqrt(mean((predictions .- targets).^2))
    
    println("Training RMSE: $(round(train_error, digits=6))\n")
    return train_error
end


# ============================================================================
#  LIQUID STATE MACHINE (LSM)
# ============================================================================

"""
    LiquidStateMachine

Alias for LIFNetwork configured as reservoir with fixed recurrent weights.

Usage:
lsm = LiquidStateMachine(neuron, 200, 10, 2;
connectivity="random", p_conn=0.1,
spectral_radius=0.9, seed=42)
train_readout!(lsm, X_train, y_train) # Train only W_out

"""
LiquidStateMachine = LIFNetwork

"""
    predict(net::LIFNetwork, input::Matrix{Float64}; feature="spike_count")

Make prediction using trained network.
"""
function predict(net::LIFNetwork, input::Matrix{Float64}; feature="spike_count")
    V_trace, spike_trains = simulate(net, input, size(input, 2) * net.neuron.dt)
    
    if feature == "spike_count"
        features = sum(spike_trains, dims=2)[:, 1]
    elseif feature == "final_voltage"
        features = V_trace[:, end]
    elseif feature == "mean_voltage"
        features = mean(V_trace, dims=2)[:, 1]
    end
    
    return net.W_out * features
end


# ============================================================================
#  ABLATION STUDY FRAMEWORK
# ============================================================================

struct AblationResult
    param_name::String
    param_value::Any
    train_error::Float64
    test_error::Float64
    spectral_radius::Float64
    n_connections::Int
end

"""
    run_ablation_study(param_name::String, param_values::Vector,
                      X_train, y_train, X_test, y_test;
                      base_params...)

Run ablation study by varying a single parameter.

Parameters: "spectral_radius", "connectivity", "N", "input_scaling", "connectivity_pattern"

# Example
results = run_ablation_study("spectral_radius", [0.5, 0.7, 0.9, 1.1],
X_train, y_train, X_test, y_test)
"""
function run_ablation_study(param_name::String, param_values::Vector,
                           X_train, y_train, X_test, y_test;
                           N=200, N_input=10, N_output=2, p_conn=0.1,
                           spectral_radius=0.9, w_scale=0.5, connectivity="random",
                           feature="spike_count", λ=0.01, seed=42)
    
    println("\n" * "="^70)
    println("ABLATION STUDY: $param_name")
    println("="^70)
    
    results = AblationResult[]
    neuron = LIFNeuron()
    
    for (idx, val) in enumerate(param_values)
        println("\nTest $idx/$(length(param_values)): $param_name = $val")
        
        # Set parameter
        current_N = param_name == "N" ? val : N
        current_p_conn = param_name == "connectivity" ? val : p_conn
        current_spectral = param_name == "spectral_radius" ? val : spectral_radius
        current_w_scale = param_name == "input_scaling" ? val : w_scale
        current_pattern = param_name == "connectivity_pattern" ? val : connectivity
        
        # Create network
        net = LIFNetwork(neuron, current_N, N_input, N_output;
                        connectivity=current_pattern, p_conn=current_p_conn,
                        spectral_radius=current_spectral, w_scale=current_w_scale, seed=seed)
        
        # Train
        train_error = train_readout!(net, X_train, y_train; λ=λ, feature=feature)
        
        # Test
        test_predictions = [predict(net, X; feature=feature) for X in X_test]
        test_pred_matrix = hcat(test_predictions...)
        test_target_matrix = hcat(y_test...)
        test_error = sqrt(mean((test_pred_matrix .- test_target_matrix).^2))
        
        # Metrics
        eigenvalues = eigvals(net.W)
        actual_spectral = maximum(abs.(eigenvalues))
        n_connections = sum(net.W .!= 0)
        
        println("  Train RMSE: $(round(train_error, digits=4))")
        println("  Test RMSE: $(round(test_error, digits=4))")
        println("  Spectral Radius: $(round(actual_spectral, digits=4))")
        println("  Connections: $n_connections")
        
        push!(results, AblationResult(param_name, val, train_error, test_error,
                                     actual_spectral, n_connections))
    end
    
    print_ablation_summary(results)
    return results
end

function print_ablation_summary(results::Vector{AblationResult})
    println("\n" * "="^70)
    println("SUMMARY: $(results[1].param_name)")
    println("="^70)
    
    println("┌" * "─"^15 * "┬" * "─"^12 * "┬" * "─"^12 * "┬" * "─"^12 * "┬" * "─"^10 * "┐")
    println("│ Parameter Val │ Train RMSE │  Test RMSE │ Spec Rad │  Conns   │")
    println("├" * "─"^15 * "┼" * "─"^12 * "┼" * "─"^12 * "┼" * "─"^12 * "┼" * "─"^10 * "┤")
    
    for r in results
        val_str = string(r.param_value)
        val_str = length(val_str) > 13 ? val_str[1:10]*"..." : val_str
        println("│ $(rpad(val_str, 14))│ $(lpad(round(r.train_error, digits=4), 11))│ " *
                "$(lpad(round(r.test_error, digits=4), 11))│ " *
                "$(lpad(round(r.spectral_radius, digits=4), 11))│ $(lpad(r.n_connections, 9))│")
    end
    
    println("└" * "─"^15 * "┴" * "─"^12 * "┴" * "─"^12 * "┴" * "─"^12 * "┴" * "─"^10 * "┘")
    
    best_idx = argmin([r.test_error for r in results])
    best = results[best_idx]
    println("\nBest: $(best.param_name) = $(best.param_value), Test RMSE = $(round(best.test_error, digits=4))\n")
end


# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

function network_stats(net::LIFNetwork)
    n_conn = sum(net.W .!= 0)
    density = n_conn / (net.N * net.N)
    mean_w = n_conn > 0 ? mean(abs.(net.W[net.W.!=0])) : 0.0
    
    eigenvalues = eigvals(net.W)
    spectral = maximum(abs.(eigenvalues))
    
    println("\n" * "="^60)
    println("Network Statistics")
    println("="^60)
    println("Neurons: $(net.N)")
    println("Input channels: $(size(net.W_in, 2))")
    println("Output channels: $(size(net.W_out, 1))")
    println("Connections: $n_conn ($(round(density*100, digits=2))% density)")
    println("Mean |weight|: $(round(mean_w, digits=4))")
    println("Spectral radius: $(round(spectral, digits=4))")
    println("="^60 * "\n")
end

function firing_rate(spikes::AbstractMatrix{Bool}, dt::Float64)
    (N, T_time) = size(spikes)
    T_max = T_time * dt
    spike_counts = sum(spikes, dims=2)[:, 1]
    return spike_counts ./ T_max
end


# ============================================================================
#  COMPLETE EXAMPLES
# ============================================================================

function example_basic_simulation()
    println("\n" * "="^60)
    println("EXAMPLE 1: Basic LIF Simulation")
    println("="^60)
    
    neuron = LIFNeuron()
    net = LIFNetwork(neuron, 100, 10, 2; connectivity="random", p_conn=0.1, seed=42)
    network_stats(net)
    
    T, n_steps = 100.0, 1000
    I_input = randn(10, n_steps) .* 3.0
    V_trace, spike_trains = simulate(net, I_input, T)
    
    rates = firing_rate(spike_trains, neuron.dt)
    println("Mean firing rate: $(round(mean(rates), digits=2)) Hz")
    println("Total spikes: $(sum(spike_trains))")
    
    return net, V_trace, spike_trains
end

function example_lsm()
    println("\n" * "="^60)
    println("EXAMPLE 2: Liquid State Machine")
    println("="^60)
    
    neuron = LIFNeuron()
    lsm = LiquidStateMachine(neuron, 200, 10, 2;
                            connectivity="random", p_conn=0.1, 
                            spectral_radius=0.9, seed=42)
    network_stats(lsm)
    
    n_train, n_test = 50, 20
    T, n_steps = 100.0, 1000
    
    X_train = [randn(10, n_steps) .* 2.0 for _ in 1:n_train]
    y_train = [[sum(X[1:2, 1]), sum(X[3:4, 1])] for X in X_train]
    
    X_test = [randn(10, n_steps) .* 2.0 for _ in 1:n_test]
    y_test = [[sum(X[1:2, 1]), sum(X[3:4, 1])] for X in X_test]
    
    train_readout!(lsm, X_train, y_train; λ=0.01)
    
    println("Testing...")
    test_errors = [sqrt(sum((predict(lsm, X) .- y).^2)) for (X, y) in zip(X_test, y_test)]
    println("Mean test error: $(round(mean(test_errors), digits=4))")
    
    return lsm
end

function example_surrogate_training()
    println("\n" * "="^60)
    println("EXAMPLE 3: Surrogate Gradient Training")
    println("="^60)
    
    neuron = LIFNeuron()
    net = LIFNetwork(neuron, 100, 10, 2; connectivity="random", p_conn=0.15, seed=42)
    
    n_samples, T, n_steps = 30, 100.0, 1000
    X_train = [randn(10, n_steps) .* 3.0 for _ in 1:n_samples]
    y_train = [randn(2) for _ in 1:n_samples]
    
    train_surrogate_gradient!(net, X_train, y_train; epochs=50, η=0.001)
    
    X_test = randn(10, n_steps) .* 3.0
    _, spike_trains = simulate(net, X_test, T)
    output = net.W_out * sum(spike_trains, dims=2)[:, 1]
    println("Test output: $(round.(output, digits=3))")
    
    return net
end

function example_ablation_spectral_radius()
    println("\n" * "="^60)
    println("EXAMPLE 4: Spectral Radius Ablation")
    println("="^60)
    
    n_train, n_test = 50, 20
    T, n_steps = 100.0, 1000
    
    X_train = [randn(10, n_steps) .* 2.0 for _ in 1:n_train]
    y_train = [[sum(X[1:3, 1])] for X in X_train]
    
    X_test = [randn(10, n_steps) .* 2.0 for _ in 1:n_test]
    y_test = [[sum(X[1:3, 1])] for X in X_test]
    
    results = run_ablation_study("spectral_radius", [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
                                X_train, y_train, X_test, y_test; N_output=1)
    
    return results
end

function example_ablation_connectivity()
    println("\n" * "="^60)
    println("EXAMPLE 5: Connectivity Ablation")
    println("="^60)
    
    n_train, n_test = 50, 20
    T, n_steps = 100.0, 1000
    
    X_train = [randn(10, n_steps) .* 2.0 for _ in 1:n_train]
    y_train = [[sum(X[1:3, 1])] for X in X_train]
    
    X_test = [randn(10, n_steps) .* 2.0 for _ in 1:n_test]
    y_test = [[sum(X[1:3, 1])] for X in X_test]
    
    results = run_ablation_study("connectivity", [0.05, 0.1, 0.15, 0.2, 0.25],
                                X_train, y_train, X_test, y_test; N_output=1)
    
    return results
end

function example_ablation_topology()
    println("\n" * "="^60)
    println("EXAMPLE 6: Topology Ablation")
    println("="^60)
    
    n_train, n_test = 50, 20
    T, n_steps = 100.0, 1000
    
    X_train = [randn(10, n_steps) .* 2.0 for _ in 1:n_train]
    y_train = [[sum(X[1:3, 1])] for X in X_train]
    
    X_test = [randn(10, n_steps) .* 2.0 for _ in 1:n_test]
    y_test = [[sum(X[1:3, 1])] for X in X_test]
    
    patterns = ["random", "small_world", "ring", "clustered"]
    results = run_ablation_study("connectivity_pattern", patterns,
                                X_train, y_train, X_test, y_test; N_output=1)
    
    return results
end

function run_all_examples()
    println("\n" * "█"^60)
    println("█ LIF.jl - COMPLETE FRAMEWORK EXAMPLES")
    println("█"^60)
    
    example_basic_simulation()
    example_lsm()
    example_surrogate_training()
    example_ablation_spectral_radius()
    example_ablation_connectivity()
    example_ablation_topology()
    
    println("\n" * "█"^60)
    println("█ ALL EXAMPLES COMPLETED!")
    println("█"^60 * "\n")
end


# ============================================================================
#  ENTRY POINT
# ============================================================================

# Run with:
# include("LIF.jl");
run_all_examples()
