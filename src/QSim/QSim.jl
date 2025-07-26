module QSim

using LinearAlgebra
using Graphs
using SparseArrays
using Statistics
using ITensors

export QuantumCircuit, StateVectorRep, TensorNetworkRep, create_quantum_simulator, add_gate!, 
       run_circuit, compute_unitary, get_measurement_probabilities, 
       print_state_probabilities, measure_z, measure_x, measure_y,
       linear_circuit, lattice_2d_circuit,
       circuit_gate_count, total_gate_count, circuit_depth, circuit_width,
       two_qubit_gate_count, single_qubit_gate_count, circuit_layers,
       print_circuit_summary, circuit_parallelism, critical_path_analysis,
       estimate_execution_time, create_bell_circuit, create_ghz_circuit,
       display_state,create_qft_circuit

# ====================================================================
# QUANTUM STATE REPRESENTATIONS
# ====================================================================

abstract type AbstractQuantumState end

mutable struct StateVectorRep <: AbstractQuantumState
    data::Vector{ComplexF64}
    n_qubits::Int
end

# Original-style tensor network (dense vector with ITensor indices)
mutable struct TensorNetworkRep <: AbstractQuantumState
    state_vector::Vector{ComplexF64}
    sites::Vector{Index}
    graph::SimpleGraph{Int64}
    n_qubits::Int
end

mutable struct QuantumCircuit
    n_qubits::Int
    state::AbstractQuantumState
    gates::Vector{Dict{Symbol,Any}}
    graph::SimpleGraph{Int}
    backend::Symbol  # :statevector or :tensornetwork
end

# ====================================================================
# GATE MATRICES AND CONSTANTS
# ====================================================================

const I2 = ComplexF64[1 0; 0 1]
const X = ComplexF64[0 1; 1 0]
const Y = ComplexF64[0 -im; im 0]
const Z = ComplexF64[1 0; 0 -1]
const H = (1/sqrt(2)) * ComplexF64[1 1; 1 -1]
const S = ComplexF64[1 0; 0 im]
const T = ComplexF64[1 0; 0 exp(im*π/4)]

# Pauli projectors
const P0 = ComplexF64[1 0; 0 0]
const P1 = ComplexF64[0 0; 0 1]

# Gate caches for performance
const RX_CACHE = Dict{Float64, Matrix{ComplexF64}}()
const RY_CACHE = Dict{Float64, Matrix{ComplexF64}}()
const RZ_CACHE = Dict{Float64, Matrix{ComplexF64}}()

function rx_matrix(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RX_CACHE, theta_f64)
        return RX_CACHE[theta_f64]
    end
    c = cos(theta_f64/2)
    s = sin(theta_f64/2)
    gate = ComplexF64[c -im*s; -im*s c]
    RX_CACHE[theta_f64] = gate
    return gate
end

function ry_matrix(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RY_CACHE, theta_f64)
        return RY_CACHE[theta_f64]
    end
    c = cos(theta_f64/2)
    s = sin(theta_f64/2)
    gate = ComplexF64[c -s; s c]
    RY_CACHE[theta_f64] = gate
    return gate
end

function rz_matrix(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RZ_CACHE, theta_f64)
        return RZ_CACHE[theta_f64]
    end
    gate = ComplexF64[exp(-im*theta_f64/2) 0; 0 exp(im*theta_f64/2)]
    RZ_CACHE[theta_f64] = gate
    return gate
end

# ====================================================================
# STATE INITIALIZATION
# ====================================================================

function create_quantum_simulator(n_qubits::Int;
                                  graph::SimpleGraph=path_graph(n_qubits),
                                  backend::Symbol=:auto,
                                  initial_state::String="zero")
    
    # Automatic backend selection (simplified)
    if backend == :auto
        backend = n_qubits <= 15 ? :statevector : :tensornetwork
    end
    
    if backend == :statevector
        dim = 1 << n_qubits
        if initial_state == "zero"
            sv = zeros(ComplexF64, dim)
            sv[1] = 1.0
        elseif initial_state == "plus"
            sv = ones(ComplexF64, dim) / sqrt(dim)
        else
            throw(ArgumentError("initial_state must be 'zero' or 'plus'"))
        end
        return StateVectorRep(sv, n_qubits)
        
    elseif backend == :tensornetwork
        # Original-style: dense vector with ITensor indices
        sites = [Index(2, "Site,n=$i") for i in 1:n_qubits]
        dim = 1 << n_qubits
        
        if initial_state == "zero"
            sv = zeros(ComplexF64, dim)
            sv[1] = 1.0
        elseif initial_state == "plus"
            sv = ones(ComplexF64, dim) / sqrt(dim)
        else
            throw(ArgumentError("initial_state must be 'zero' or 'plus'"))
        end
        
        return TensorNetworkRep(sv, sites, graph, n_qubits)
    else
        throw(ArgumentError("backend must be :statevector, :tensornetwork, or :auto"))
    end
end

function QuantumCircuit(n_qubits::Int; 
                        graph::SimpleGraph=path_graph(n_qubits), 
                        backend::Symbol=:auto,
                        initial_state::String="zero")
    # Determine actual backend for storage
    actual_backend = backend == :auto ? (n_qubits <= 15 ? :statevector : :tensornetwork) : backend
    state = create_quantum_simulator(n_qubits, graph=graph, backend=backend, initial_state=initial_state)
    QuantumCircuit(n_qubits, state, Vector{Dict{Symbol,Any}}(), graph, actual_backend)
end

# ====================================================================
# OPTIMIZED STATE VECTOR OPERATIONS
# ====================================================================

function kronN(n::Int, op::Matrix{ComplexF64}, target::Int)
    """Apply single-qubit gate to target position in n-qubit system"""
    factors = Vector{Matrix{ComplexF64}}(undef, n)
    for i in 1:n
        factors[i] = (i == target) ? op : I2
    end
    return reduce(kron, factors)
end

function apply_single_gate_inplace!(state_vec::Vector{ComplexF64}, gate_matrix::Matrix{ComplexF64}, 
                                   target::Int, n_qubits::Int)
    """Apply single-qubit gate efficiently in-place using bit manipulation"""
    mask = 1 << (target - 1)
    step = 1 << target
    
    @inbounds for i in 0:step:(length(state_vec)-1)
        for j in i:(i + mask - 1)
            j0 = j + 1
            j1 = j + mask + 1
            
            if j1 <= length(state_vec)
                amp0 = state_vec[j0]
                amp1 = state_vec[j1]
                
                state_vec[j0] = gate_matrix[1,1] * amp0 + gate_matrix[1,2] * amp1
                state_vec[j1] = gate_matrix[2,1] * amp0 + gate_matrix[2,2] * amp1
            end
        end
    end
end

function apply_cnot_inplace!(state_vec::Vector{ComplexF64}, control::Int, target::Int, n_qubits::Int)
    """Apply CNOT gate efficiently in-place using XOR operations"""
    ctrl_mask = 1 << (control - 1)
    targ_mask = 1 << (target - 1)
    
    @inbounds for i in 0:(length(state_vec)-1)
        if (i & ctrl_mask) != 0
            j = xor(i, targ_mask)
            if j > i
                state_vec[i+1], state_vec[j+1] = state_vec[j+1], state_vec[i+1]
            end
        end
    end
end

function apply_controlled_gate_inplace!(state_vec::Vector{ComplexF64}, gate_matrix::Matrix{ComplexF64},
                                       control::Int, target::Int, n_qubits::Int)
    """Apply controlled single-qubit gate efficiently"""
    ctrl_mask = 1 << (control - 1)
    targ_mask = 1 << (target - 1)
    step = 1 << target
    
    @inbounds for i in 0:(length(state_vec)-1)
        if (i & ctrl_mask) != 0  # Control is |1⟩
            # Apply gate to target
            target_bit = (i & targ_mask) >> (target - 1)
            if target_bit == 0
                j = i | targ_mask  # Set target to |1⟩
                if j < length(state_vec)
                    amp0 = state_vec[i+1]
                    amp1 = state_vec[j+1]
                    
                    state_vec[i+1] = gate_matrix[1,1] * amp0 + gate_matrix[1,2] * amp1
                    state_vec[j+1] = gate_matrix[2,1] * amp0 + gate_matrix[2,2] * amp1
                end
            end
        end
    end
end

function apply_cp_statevector!(state::StateVectorRep, control::Int, target::Int, angle::Float64)
    """Apply controlled phase gate to statevector representation"""
    n = state.n_qubits
    control_mask = 1 << (control - 1)
    target_mask = 1 << (target - 1)
    
    # Apply phase only when both control and target qubits are |1⟩
    for i in 0:(2^n - 1)
        if ((i & control_mask) != 0) && ((i & target_mask) != 0)
            state.data[i+1] *= exp(im * angle)
        end
    end
    
    return state
end

function apply_cz_statevector!(state::StateVectorRep, control::Int, target::Int)
    """Apply controlled-Z gate to statevector representation"""
    n = state.n_qubits
    control_mask = 1 << (control - 1)
    target_mask = 1 << (target - 1)
    
    # Apply -1 phase when both qubits are |1⟩
    for i in 0:(2^n - 1)
        if ((i & control_mask) != 0) && ((i & target_mask) != 0)
            state.data[i+1] *= -1
        end
    end
    
    return state
end

# ====================================================================
# TENSOR NETWORK OPERATIONS (ORIGINAL STYLE)
# ====================================================================

function apply_hadamard_tensor!(sim::TensorNetworkRep, qubit::Int)
    """Apply Hadamard gate using original tensor network approach"""
    if qubit < 1 || qubit > sim.n_qubits
        throw(ArgumentError("Qubit index $qubit out of range [1, $(sim.n_qubits)]"))
    end
    
    new_state = zeros(ComplexF64, length(sim.state_vector))
    inv_sqrt2 = 1.0 / sqrt(2)
    
    @inbounds for i in 0:(2^sim.n_qubits - 1)
        bit_string = digits(i, base=2, pad=sim.n_qubits)
        
        bit_string_0 = copy(bit_string)
        bit_string_1 = copy(bit_string)
        bit_string_0[qubit] = 0
        bit_string_1[qubit] = 1
        
        idx_0 = sum(bit_string_0 .* (2 .^ (0:(sim.n_qubits-1)))) + 1
        idx_1 = sum(bit_string_1 .* (2 .^ (0:(sim.n_qubits-1)))) + 1
        
        if bit_string[qubit] == 0
            new_state[idx_0] += sim.state_vector[i+1] * inv_sqrt2
            new_state[idx_1] += sim.state_vector[i+1] * inv_sqrt2
        else
            new_state[idx_0] += sim.state_vector[i+1] * inv_sqrt2
            new_state[idx_1] -= sim.state_vector[i+1] * inv_sqrt2
        end
    end
    
    sim.state_vector = new_state
    return sim
end

function apply_single_gate_tensor!(sim::TensorNetworkRep, gate_matrix::Matrix{ComplexF64}, qubit::Int)
    """Apply arbitrary single-qubit gate using tensor network approach"""
    if qubit < 1 || qubit > sim.n_qubits
        throw(ArgumentError("Qubit index $qubit out of range [1, $(sim.n_qubits)]"))
    end
    
    new_state = zeros(ComplexF64, length(sim.state_vector))
    
    @inbounds for i in 0:(2^sim.n_qubits - 1)
        bit_string = digits(i, base=2, pad=sim.n_qubits)
        
        bit_string_0 = copy(bit_string)
        bit_string_1 = copy(bit_string)
        bit_string_0[qubit] = 0
        bit_string_1[qubit] = 1
        
        idx_0 = sum(bit_string_0 .* (2 .^ (0:(sim.n_qubits-1)))) + 1
        idx_1 = sum(bit_string_1 .* (2 .^ (0:(sim.n_qubits-1)))) + 1
        
        if bit_string[qubit] == 0
            new_state[idx_0] += gate_matrix[1,1] * sim.state_vector[i+1]
            new_state[idx_1] += gate_matrix[2,1] * sim.state_vector[i+1]
        else
            new_state[idx_0] += gate_matrix[1,2] * sim.state_vector[i+1]
            new_state[idx_1] += gate_matrix[2,2] * sim.state_vector[i+1]
        end
    end
    
    sim.state_vector = new_state
    return sim
end

function apply_cnot_tensor!(sim::TensorNetworkRep, control::Int, target::Int)
    """Apply CNOT gate using tensor network approach"""
    new_state = zeros(ComplexF64, length(sim.state_vector))
    
    @inbounds for i in 0:(2^sim.n_qubits - 1)
        bit_string = digits(i, base=2, pad=sim.n_qubits)
        output_bit_string = copy(bit_string)
        
        if bit_string[control] == 1
            output_bit_string[target] = 1 - bit_string[target]
        end
        
        output_idx = sum(output_bit_string .* (2 .^ (0:(sim.n_qubits-1)))) + 1
        new_state[output_idx] += sim.state_vector[i+1]
    end
    
    sim.state_vector = new_state
    return sim
end

function apply_cp_tensor!(state, control, target, angle)
    """Apply controlled phase gate using tensor network approach"""
    # Get dimensions
    dims = size(state.tensor)
    
    # Apply controlled phase operation
    for idx in CartesianIndices(dims)
        indices = Tuple(idx)
        # If control and target qubits are both 1 (indices start at 1, so 2 means |1⟩)
        if indices[control] == 2 && indices[target] == 2
            # Apply phase
            state.tensor[idx] *= exp(im * angle)
        end
    end
    
    return state
end

# ====================================================================
# UNIFIED GATE APPLICATION INTERFACE
# ====================================================================

function add_gate!(circuit::QuantumCircuit, gate_type::Symbol; 
                   targets::Vector{Int}, controls::Vector{Int}=Int[], 
                   params::Vector{<:Real}=Real[])
    # Validate inputs
    all(t -> 1 <= t <= circuit.n_qubits, targets) || error("Target qubits out of range")
    all(c -> 1 <= c <= circuit.n_qubits, controls) || error("Control qubits out of range")
    
    gate_dict = Dict(
        :gate => gate_type,
        :targets => targets,
        :controls => controls,
        :params => params
    )
    push!(circuit.gates, gate_dict)
    return circuit
end

function apply_gate!(state::StateVectorRep, gate::Dict{Symbol,Any})
    n = state.n_qubits
    gtype = gate[:gate]
    targets = gate[:targets]
    controls = gate[:controls]
    params = gate[:params]

    if gtype == :h
        apply_single_gate_inplace!(state.data, H, targets[1], n)
    elseif gtype == :x
        apply_single_gate_inplace!(state.data, X, targets[1], n)
    elseif gtype == :y
        apply_single_gate_inplace!(state.data, Y, targets[1], n)
    elseif gtype == :z
        apply_single_gate_inplace!(state.data, Z, targets[1], n)
    elseif gtype == :s
        apply_single_gate_inplace!(state.data, S, targets[1], n)
    elseif gtype == :t
        apply_single_gate_inplace!(state.data, T, targets[1], n)
    elseif gtype == :rx
        apply_single_gate_inplace!(state.data, rx_matrix(params[1]), targets[1], n)
    elseif gtype == :ry
        apply_single_gate_inplace!(state.data, ry_matrix(params[1]), targets[1], n)
    elseif gtype == :rz
        apply_single_gate_inplace!(state.data, rz_matrix(params[1]), targets[1], n)
    elseif gtype == :cnot
        apply_cnot_inplace!(state.data, controls[1], targets[1], n)
    elseif gtype == :cp
        control = controls[1]
        target = targets[1]
        angle = params[1]
        apply_cp_statevector!(state, control, target, angle)
    
    elseif gtype == :cz
        control = controls[1]
        target = targets[1]
        apply_cz_statevector!(state, control, target)
    elseif gtype == :swap
        t1, t2 = targets[1], targets[2]
        apply_cnot_inplace!(state.data, t1, t2, n)
        apply_cnot_inplace!(state.data, t2, t1, n)
        apply_cnot_inplace!(state.data, t1, t2, n)
    else
        error("Unsupported gate: $gtype")
    end
    return state
end

function apply_cp_tensor!(sim::TensorNetworkRep, control::Int, target::Int, angle::Real)
    """Apply controlled phase gate using tensor network approach"""
    new_state = copy(sim.state_vector)
    
    @inbounds for i in 0:(2^sim.n_qubits - 1)
        bit_string = digits(i, base=2, pad=sim.n_qubits)
        
        # If both control and target qubits are 1, apply phase
        if bit_string[control] == 1 && bit_string[target] == 1
            new_state[i+1] *= exp(im * angle)
        end
    end
    
    sim.state_vector = new_state
    return sim
end

function apply_gate!(state::TensorNetworkRep, gate::Dict{Symbol,Any})
    gtype = gate[:gate]
    targets = gate[:targets]
    controls = gate[:controls]
    params = gate[:params]

    if gtype == :h
        apply_hadamard_tensor!(state, targets[1])
    elseif gtype == :x
        apply_single_gate_tensor!(state, X, targets[1])
    elseif gtype == :y
        apply_single_gate_tensor!(state, Y, targets[1])
    elseif gtype == :z
        apply_single_gate_tensor!(state, Z, targets[1])
    elseif gtype == :s
        apply_single_gate_tensor!(state, S, targets[1])
    elseif gtype == :t
        apply_single_gate_tensor!(state, T, targets[1])
    elseif gtype == :rx
        apply_single_gate_tensor!(state, rx_matrix(params[1]), targets[1])
    elseif gtype == :ry
        apply_single_gate_tensor!(state, ry_matrix(params[1]), targets[1])
    elseif gtype == :rz
        apply_single_gate_tensor!(state, rz_matrix(params[1]), targets[1])
    elseif gtype == :cnot
        apply_cnot_tensor!(state, controls[1], targets[1])
    elseif gtype == :cp
        control = controls[1]
        target = targets[1]
        angle = params[1]
        apply_cp_tensor!(state, control, target, angle)
    

    elseif gtype == :swap
        # Implement SWAP as three CNOTs
        t1, t2 = targets[1], targets[2]
        apply_cnot_tensor!(state, t1, t2)
        apply_cnot_tensor!(state, t2, t1)
        apply_cnot_tensor!(state, t1, t2)
    else
        error("Unsupported tensor network gate: $gtype")
    end
    return state
end

# ====================================================================
# CIRCUIT EXECUTION
# ====================================================================

function run_circuit(circuit::QuantumCircuit)
    """Execute all gates in the circuit"""
    for gate in circuit.gates
        apply_gate!(circuit.state, gate)
    end
    return circuit.state
end

function run_circuit!(circuit::QuantumCircuit)
    """Execute all gates in the circuit (in-place version)"""
    run_circuit(circuit)
    return circuit
end

# ====================================================================
# UNITARY COMPUTATION
# ====================================================================

function compute_unitary(circuit::QuantumCircuit)
    """Compute the full unitary matrix of the circuit"""
    n = circuit.n_qubits
    if n > 15
        @warn "Computing unitary for $n qubits requires $(2^(2*n)) complex numbers ($(2^(2*n)*16) bytes)"
    end
    
    U = Matrix{ComplexF64}(I, 1<<n, 1<<n)
    
    for gate in circuit.gates
        gate_matrix = get_gate_unitary(gate, n)
        U = gate_matrix * U
    end
    return U
end

function get_gate_unitary(gate::Dict{Symbol,Any}, n_qubits::Int)
    """Get the full unitary matrix for a gate"""
    gtype = gate[:gate]
    targets = gate[:targets]
    controls = gate[:controls]
    params = gate[:params]
    
    if gtype == :h
        return kronN(n_qubits, H, targets[1])
    elseif gtype == :x
        return kronN(n_qubits, X, targets[1])
    elseif gtype == :y
        return kronN(n_qubits, Y, targets[1])
    elseif gtype == :z
        return kronN(n_qubits, Z, targets[1])
    elseif gtype == :s
        return kronN(n_qubits, S, targets[1])
    elseif gtype == :t
        return kronN(n_qubits, T, targets[1])
    elseif gtype == :rx
        return kronN(n_qubits, rx_matrix(params[1]), targets[1])
    elseif gtype == :ry
        return kronN(n_qubits, ry_matrix(params[1]), targets[1])
    elseif gtype == :rz
        return kronN(n_qubits, rz_matrix(params[1]), targets[1])
    elseif gtype == :cnot
        return build_controlled_gate(X, controls[1], targets[1], n_qubits)
    elseif gtype == :cz
        return build_controlled_gate(Z, controls[1], targets[1], n_qubits)
    elseif gtype == :swap
        return build_swap_unitary(targets[1], targets[2], n_qubits)
    elseif gate[:gate] == :cp
        control = gate[:controls][1]
        target = gate[:targets][1]
        angle = gate[:params][1]
        return build_cp_unitary(control, target, n_qubits, angle)
    else
        error("Unsupported gate for unitary: $gtype")
    end
end

function build_controlled_gate(gate::Matrix{ComplexF64}, control::Int, target::Int, n_qubits::Int)
    """Build controlled gate matrix efficiently"""
    dim = 1 << n_qubits
    result = zeros(ComplexF64, dim, dim)
    
    @inbounds for i in 0:(dim-1)
        bits = digits(i, base=2, pad=n_qubits)
        
        if bits[control] == 0
            result[i+1, i+1] = 1.0
        else
            output_bits = copy(bits)
            if gate == X
                output_bits[target] = 1 - output_bits[target]
            elseif gate == Z && bits[target] == 1
                result[i+1, i+1] = -1.0
                continue
            end
            
            j = sum(output_bits .* (2 .^ (0:n_qubits-1)))
            if gate == X
                result[j+1, i+1] = 1.0
            else
                result[i+1, i+1] = 1.0
            end
        end
    end
    return result
end

function build_cp_unitary(control::Int, target::Int, n_qubits::Int, angle::Float64)
    """Build controlled phase gate unitary matrix
    
    Args:
        control: control qubit index
        target: target qubit index
        n_qubits: total number of qubits
        angle: rotation angle in radians
    
    Returns:
        ComplexF64 matrix for the controlled phase gate
    """
    dim = 1 << n_qubits
    result = Matrix{ComplexF64}(I, dim, dim)
    
    # Apply phase only when both control and target qubits are 1
    @inbounds for i in 0:(dim-1)
        bits = digits(i, base=2, pad=n_qubits)
        # If both control and target are 1
        if bits[control] == 1 && bits[target] == 1
            # Apply phase
            result[i+1, i+1] = exp(im * angle)
        end
    end
    
    return result
end

function build_swap_unitary(q1::Int, q2::Int, n_qubits::Int)
    """Build SWAP gate unitary"""
    dim = 1 << n_qubits
    result = zeros(ComplexF64, dim, dim)
    
    @inbounds for i in 0:(dim-1)
        bits = digits(i, base=2, pad=n_qubits)
        output_bits = copy(bits)
        output_bits[q1], output_bits[q2] = bits[q2], bits[q1]
        j = sum(output_bits .* (2 .^ (0:n_qubits-1)))
        result[j+1, i+1] = 1.0
    end
    return result
end

# ====================================================================
# MEASUREMENT AND ANALYSIS
# ====================================================================

function get_measurement_probabilities(state::StateVectorRep)
    return abs2.(state.data)
end

function get_measurement_probabilities(state::TensorNetworkRep)
    return abs2.(state.state_vector)
end

function measure_z(state::AbstractQuantumState, target::Int)
    """Measure single qubit in Z basis"""
    if isa(state, StateVectorRep)
        data = state.data
    else  # TensorNetworkRep
        data = state.state_vector
    end
    
    n = state.n_qubits
    probs = abs2.(data)
    p0 = 0.0
    p1 = 0.0
    
    mask = 1 << (target - 1)
    @inbounds for i in 0:(length(probs)-1)
        if (i & mask) == 0
            p0 += probs[i+1]
        else
            p1 += probs[i+1]
        end
    end
    return (p0, p1)
end

function measure_x(state::AbstractQuantumState, target::Int)
    """Measure single qubit in X basis"""
    if isa(state, StateVectorRep)
        state_copy = StateVectorRep(copy(state.data), state.n_qubits)
        apply_single_gate_inplace!(state_copy.data, H, target, state.n_qubits)
    else  # TensorNetworkRep
        state_copy = TensorNetworkRep(copy(state.state_vector), state.sites, state.graph, state.n_qubits)
        apply_hadamard_tensor!(state_copy, target)
    end
    return measure_z(state_copy, target)
end

function measure_y(state::AbstractQuantumState, target::Int)
    """Measure single qubit in Y basis"""
    s_dag = ComplexF64[1 0; 0 -im]
    
    if isa(state, StateVectorRep)
        state_copy = StateVectorRep(copy(state.data), state.n_qubits)
        apply_single_gate_inplace!(state_copy.data, s_dag, target, state.n_qubits)
        apply_single_gate_inplace!(state_copy.data, H, target, state.n_qubits)
    else  # TensorNetworkRep
        state_copy = TensorNetworkRep(copy(state.state_vector), state.sites, state.graph, state.n_qubits)
        apply_single_gate_tensor!(state_copy, s_dag, target)
        apply_hadamard_tensor!(state_copy, target)
    end
    return measure_z(state_copy, target)
end

function print_state_probabilities(state::AbstractQuantumState; threshold::Real=1e-6)
    """Print non-zero measurement probabilities"""
    n = state.n_qubits
    probs = get_measurement_probabilities(state)
    
    println("=== Measurement Probabilities ===")
    total_shown = 0.0
    
    for (i, p) in enumerate(probs)
        if p > threshold
            bits = digits(i-1, base=2, pad=n)
            basis_state = join(reverse(bits))
            println("|$basis_state⟩: $(round(p, digits=6))")
            total_shown += p
        end
    end
    
    println("Total probability shown: $(round(total_shown, digits=6))")
    println("===============================")
end

function display_state(state::AbstractQuantumState)
    """Enhanced state display function"""
    if isa(state, StateVectorRep)
        println("State Vector Backend ($(state.n_qubits) qubits):")
        print_state_probabilities(state)
    elseif isa(state, TensorNetworkRep)
        println("Tensor Network Backend ($(state.n_qubits) qubits):")
        println("ITensor sites: $(length(state.sites))")
        print_state_probabilities(state)
    end
end

# ====================================================================
# CIRCUIT ANALYSIS UTILITIES
# ====================================================================

function circuit_gate_count(circuit::QuantumCircuit)
    """Count gates by type"""
    counts = Dict{Symbol, Int}()
    for gate in circuit.gates
        gtype = gate[:gate]
        counts[gtype] = get(counts, gtype, 0) + 1
    end
    return counts
end

function total_gate_count(circuit::QuantumCircuit)
    """Total number of gates in circuit"""
    return length(circuit.gates)
end

function circuit_depth(circuit::QuantumCircuit)
    """Calculate true circuit depth considering gate parallelization"""
    if isempty(circuit.gates)
        return 0
    end
    
    n_qubits = circuit.n_qubits
    qubit_times = zeros(Int, n_qubits)
    
    for gate in circuit.gates
        involved_qubits = union(gate[:targets], gate[:controls])
        start_time = maximum(qubit_times[involved_qubits])
        
        for qubit in involved_qubits
            qubit_times[qubit] = start_time + 1
        end
    end
    
    return maximum(qubit_times)
end

function circuit_width(circuit::QuantumCircuit)
    """Number of qubits used in the circuit"""
    used_qubits = Set{Int}()
    for gate in circuit.gates
        union!(used_qubits, gate[:targets])
        union!(used_qubits, gate[:controls])
    end
    return length(used_qubits)
end

function two_qubit_gate_count(circuit::QuantumCircuit)
    """Count two-qubit gates"""
    two_qubit_gates = [:cnot, :swap, :cz, :cy, :crx, :cry, :crz]
    count = 0
    for gate in circuit.gates
        if gate[:gate] in two_qubit_gates
            count += 1
        end
    end
    return count
end

function single_qubit_gate_count(circuit::QuantumCircuit)
    """Count single-qubit gates"""
    single_qubit_gates = [:h, :x, :y, :z, :rx, :ry, :rz, :s, :t]
    count = 0
    for gate in circuit.gates
        if gate[:gate] in single_qubit_gates
            count += 1
        end
    end
    return count
end

function circuit_layers(circuit::QuantumCircuit)
    """Group gates into parallel executable layers"""
    if isempty(circuit.gates)
        return Vector{Vector{Int}}()
    end
    
    layers = Vector{Vector{Int}}()
    gate_scheduled = fill(false, length(circuit.gates))
    
    while !all(gate_scheduled)
        current_layer = Int[]
        used_qubits = Set{Int}()
        
        for (i, gate) in enumerate(circuit.gates)
            if gate_scheduled[i]
                continue
            end
            
            involved_qubits = union(gate[:targets], gate[:controls])
            
            if isempty(intersect(involved_qubits, used_qubits))
                push!(current_layer, i)
                union!(used_qubits, involved_qubits)
                gate_scheduled[i] = true
            end
        end
        
        push!(layers, current_layer)
    end
    
    return layers
end

function print_circuit_summary(circuit::QuantumCircuit)
    """Print comprehensive circuit analysis"""
    println("=" ^ 50)
    println("QUANTUM CIRCUIT SUMMARY")
    println("=" ^ 50)
    
    println("📊 Basic Metrics:")
    println("  • Backend: $(circuit.backend)")
    println("  • Total qubits: $(circuit.n_qubits)")
    println("  • Active qubits: $(circuit_width(circuit))")
    println("  • Total gates: $(total_gate_count(circuit))")
    println("  • Circuit depth: $(circuit_depth(circuit))")
    println("  • Number of layers: $(length(circuit_layers(circuit)))")
    
    println("\n🚪 Gate Breakdown:")
    gate_counts = circuit_gate_count(circuit)
    for (gate_type, count) in sort(collect(gate_counts))
        println("  • $gate_type: $count")
    end
    
    two_q_count = two_qubit_gate_count(circuit)
    one_q_count = single_qubit_gate_count(circuit)
    println("\n🔗 Gate Categories:")
    println("  • Single-qubit gates: $one_q_count")
    println("  • Two-qubit gates: $two_q_count")
    
    println("\n🌐 Connectivity:")
    println("  • Graph vertices: $(nv(circuit.graph))")
    println("  • Graph edges: $(ne(circuit.graph))")
    
    if circuit.n_qubits <= 20
        state_size = 2^circuit.n_qubits * 16
        unitary_size = 2^(2*circuit.n_qubits) * 16
        println("\n💾 Memory Requirements:")
        println("  • State vector: $(format_bytes(state_size))")
        println("  • Full unitary: $(format_bytes(unitary_size))")
    else
        println("\n⚠️  Large system: Memory usage may be significant")
    end
    
    println("=" ^ 50)
end

function format_bytes(bytes::Int)
    """Format byte count in human-readable format"""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes)
    unit_idx = 1
    
    while size >= 1024 && unit_idx < length(units)
        size /= 1024
        unit_idx += 1
    end
    
    return "$(round(size, digits=2)) $(units[unit_idx])"
end

function circuit_parallelism(circuit::QuantumCircuit)
    """Calculate average parallelism (gates per layer)"""
    layers = circuit_layers(circuit)
    if isempty(layers)
        return 0.0
    end
    
    total_gates = sum(length(layer) for layer in layers)
    return total_gates / length(layers)
end

function critical_path_analysis(circuit::QuantumCircuit)
    """Analyze critical path for each qubit"""
    n_qubits = circuit.n_qubits
    qubit_gate_counts = zeros(Int, n_qubits)
    
    for gate in circuit.gates
        involved_qubits = union(gate[:targets], gate[:controls])
        for qubit in involved_qubits
            qubit_gate_counts[qubit] += 1
        end
    end
    
    return Dict(
        :max_qubit_gates => maximum(qubit_gate_counts),
        :min_qubit_gates => minimum(qubit_gate_counts),
        :avg_qubit_gates => mean(qubit_gate_counts),
        :qubit_gate_counts => qubit_gate_counts
    )
end

function estimate_execution_time(circuit::QuantumCircuit; 
                                gate_times::Dict{Symbol,Float64}=Dict(
                                    :h => 20e-9, :x => 20e-9, :y => 20e-9, :z => 0.0,
                                    :rx => 20e-9, :ry => 20e-9, :rz => 0.0,
                                    :cnot => 200e-9, :swap => 600e-9, :cz => 200e-9
                                ))
    """Estimate circuit execution time on quantum hardware"""
    layers = circuit_layers(circuit)
    total_time = 0.0
    
    for layer in layers
        layer_time = 0.0
        for gate_idx in layer
            gate = circuit.gates[gate_idx]
            gate_type = gate[:gate]
            gate_time = get(gate_times, gate_type, 50e-9)
            layer_time = max(layer_time, gate_time)
        end
        total_time += layer_time
    end
    
    return total_time
end

# ====================================================================
# TOPOLOGY UTILITIES
# ====================================================================

function linear_circuit(n_qubits::Int)
    """Create linear chain topology"""
    return path_graph(n_qubits)
end

function lattice_2d_circuit(rows::Int, cols::Int)
    """Create 2D grid lattice topology"""
    return grid([rows, cols])
end

function all_to_all_circuit(n_qubits::Int)
    """Create fully connected topology"""
    return complete_graph(n_qubits)
end

# ====================================================================
# CONVENIENCE FUNCTIONS
# ====================================================================

function create_bell_circuit(; backend::Symbol=:auto)
    """Create Bell state preparation circuit"""
    circuit = QuantumCircuit(2, backend=backend)
    add_gate!(circuit, :h, targets=[1])
    add_gate!(circuit, :cnot, controls=[1], targets=[2])
    return circuit
end

function create_ghz_circuit(n_qubits::Int; backend::Symbol=:auto)
    """Create GHZ state preparation circuit"""
    circuit = QuantumCircuit(n_qubits, backend=backend)
    add_gate!(circuit, :h, targets=[1])
    for i in 2:n_qubits
        add_gate!(circuit, :cnot, controls=[1], targets=[i])
    end
    return circuit
end

function create_qft_circuit(n_qubits::Int; backend::Symbol=:auto)
    """Create Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(n_qubits, backend=backend)
    
    for i in 1:n_qubits
        add_gate!(circuit, :h, targets=[i])
        for j in (i+1):n_qubits
            angle = π / (2^(j-i))
            # Approximate controlled rotation with RZ and CNOT
            add_gate!(circuit, :cnot, controls=[j], targets=[i])
            add_gate!(circuit, :rz, targets=[i], params=[angle])
            add_gate!(circuit, :cnot, controls=[j], targets=[i])
        end
    end
    
    # Reverse qubit order
    for i in 1:(n_qubits÷2)
        add_gate!(circuit, :swap, targets=[i, n_qubits-i+1])
    end
    
    return circuit
end

end # module HybridQSim
