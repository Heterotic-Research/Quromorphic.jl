module Grovers
using Heterotic.QSim
using LinearAlgebra

export oracle, diffusion, create_oracle_circuit, create_diffusion_circuit, create_grovers_circuit


"""
    create_oracle_circuit(circuit, marked_states)

Create an oracle circuit that marks the specified states by flipping their phase.
The marked_states should be provided as a list of integers representing the computational basis states.
"""
function create_oracle_circuit(circuit::QuantumCircuit, marked_states::Vector{Int})
    n_qubits = circuit.n_qubits
    
    for state in marked_states
        # Convert state index to binary representation
        binary = digits(state, base=2, pad=n_qubits)
        
        # Apply X gates to all qubits where the bit is 0
        for q in 1:n_qubits
            if binary[q] == 0
                add_gate!(circuit, :x, targets=[q])
            end
        end
        
        # Apply multi-controlled Z gate
        if n_qubits == 1
            add_gate!(circuit, :z, targets=[1])
        elseif n_qubits == 2
            add_gate!(circuit, :cz, controls=[1], targets=[2])
        else
            # For n>2, apply X to all but last qubit, then use multi-controlled-X, then X again
            for q in 1:(n_qubits-1)
                add_gate!(circuit, :x, targets=[q])
            end
            
            # Multi-controlled NOT on the last qubit
            controls = collect(1:(n_qubits-1))
            add_gate!(circuit, :cnot, controls=controls, targets=[n_qubits])
            
            # Z on the last qubit
            add_gate!(circuit, :z, targets=[n_qubits])
            
            # Repeat multi-controlled NOT to restore state
            add_gate!(circuit, :cnot, controls=controls, targets=[n_qubits])
            
            # Undo X gates (handled by the outer loop below)
        end
        
        # Undo X gates
        for q in 1:n_qubits
            if binary[q] == 0
                add_gate!(circuit, :x, targets=[q])
            end
        end
    end
    
    return circuit
end

"""
    create_diffusion_circuit(circuit)

Create a diffusion circuit for Grover's algorithm.
This implements the diffusion operator 2|s⟩⟨s| - I.
"""
function create_diffusion_circuit(circuit::QuantumCircuit)
    n_qubits = circuit.n_qubits
    
    # Apply H to all qubits
    for i in 1:n_qubits
        add_gate!(circuit, :h, targets=[i])
    end
    
    # Apply X to all qubits
    for i in 1:n_qubits
        add_gate!(circuit, :x, targets=[i])
    end
    
    # Apply controlled-Z gate (phase flip on |11...1⟩ state)
    if n_qubits == 1
        add_gate!(circuit, :z, targets=[1])
    elseif n_qubits == 2
        add_gate!(circuit, :cz, controls=[1], targets=[2])
    else
        # For n>2, use a multi-controlled Z implementation
        # First apply H to the last qubit
        add_gate!(circuit, :h, targets=[n_qubits])
        
        # Apply multi-controlled-NOT with all but the last qubit as controls
        controls = collect(1:(n_qubits-1))
        add_gate!(circuit, :cnot, controls=controls, targets=[n_qubits])
        
        # Apply H to the last qubit again
        add_gate!(circuit, :h, targets=[n_qubits])
    end
    
    # Apply X to all qubits again
    for i in 1:n_qubits
        add_gate!(circuit, :x, targets=[i])
    end
    
    # Apply H to all qubits again
    for i in 1:n_qubits
        add_gate!(circuit, :h, targets=[i])
    end
    
    return circuit
end

"""
    create_grovers_circuit(n_qubits, marked_states, iterations)

Create a complete Grover's algorithm circuit for searching the marked states.
"""
function create_grovers_circuit(n_qubits::Int, marked_states::Vector{Int}, iterations::Int)
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize with Hadamard on all qubits
    for i in 1:n_qubits
        add_gate!(circuit, :h, targets=[i])
    end
    
    # Apply Grover iteration the specified number of times
    for _ in 1:iterations
        # Oracle
        create_oracle_circuit(circuit, marked_states)
        
        # Diffusion
        create_diffusion_circuit(circuit)
    end
    
    return circuit
end

end