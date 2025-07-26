module QFT

using Heterotic.QSim

export qft, iqft, dft_matrix, idft_matrix, phase_estimation

"""
    qft(n::Int) -> QuantumCircuit

Create a Quantum Fourier Transform circuit for n qubits.
The QFT transforms from the computational basis to the Fourier basis.
"""
function qft(n::Int)
    circuit = QuantumCircuit(n)
    
    # Apply QFT
    for i in 1:n
        # Hadamard gate on the current qubit
        add_gate!(circuit, :h, targets=[i])
        
        # Controlled rotations
        for j in (i+1):n
            # Phase rotation by 2π/2^(j-i+1)
            angle = 2π / (2^(j-i+1))
            add_gate!(circuit, :cp, controls=[i], targets=[j], params=[angle])
        end
    end
    
    # Swap qubits to correct the output order
    for i in 1:div(n, 2)
        add_gate!(circuit, :swap, targets=[i, n-i+1])
    end
    
    return circuit
end

"""
    iqft(n::Int) -> QuantumCircuit

Create an Inverse Quantum Fourier Transform circuit for n qubits.
The inverse QFT transforms from the Fourier basis back to the computational basis.
"""
function iqft(n::Int)
    circuit = QuantumCircuit(n)
    
    # Swap qubits first (reverse of the QFT's final swap)
    for i in 1:div(n, 2)
        add_gate!(circuit, :swap, targets=[i, n-i+1])
    end
    
    # Apply inverse QFT (reversed order and negated angles)
    for i in n:-1:1
        # Inverse controlled rotations
        for j in n:-1:(i+1)
            # Negative phase rotation by -2π/2^(j-i+1)
            angle = -2π / (2^(j-i+1))
            add_gate!(circuit, :cp, controls=[i], targets=[j], params=[angle])
        end
        
        # Hadamard gate on the current qubit
        add_gate!(circuit, :h, targets=[i])
    end
    
    return circuit
end

"""
    qft_recursive(circuit::QuantumCircuit, first::Int, last::Int) -> QuantumCircuit

Helper function to recursively construct a QFT circuit.
Applies the QFT transformation to qubits from first to last.
"""
function qft_recursive(circuit::QuantumCircuit, first::Int, last::Int)
    if first >= last
        return circuit
    end
    
    # Apply Hadamard to the first qubit
    add_gate!(circuit, :h, targets=[first])
    
    # Apply controlled rotations from second qubit onwards
    for j in (first+1):last
        k = j - first + 1
        angle = 2π / (2^k)
        add_gate!(circuit, :cp, controls=[first], targets=[j], params=[angle])
    end
    
    # Recursively apply to the remaining qubits
    return qft_recursive(circuit, first + 1, last)
end



"""
    phase_estimation(u_circuit::QuantumCircuit, precision::Int) -> QuantumCircuit

Create a Quantum Phase Estimation circuit with the given unitary and precision.
Uses QFT as a subroutine.
"""
function phase_estimation(u_circuit::QuantumCircuit, precision::Int)
    # Assuming u_circuit operates on 1 qubit
    target_qubits = u_circuit.n_qubits
    total_qubits = precision + target_qubits
    
    circuit = QuantumCircuit(total_qubits)
    
    # Initialize the first register to |+⟩ states
    for i in 1:precision
        add_gate!(circuit, :h, targets=[i])
    end
    
    # Apply controlled powers of U
    for i in 1:precision
        power = 2^(i-1)
        
        # Apply U^(2^(i-1)) controlled by qubit i
        for _ in 1:power
            # Map the controls and targets from u_circuit to the new circuit
            for gate in u_circuit.gates
                if haskey(gate, :controls)
                    controls = gate[:controls] .+ precision
                    add_gate!(circuit, gate[:gate], 
                             controls=[i; controls], 
                             targets=gate[:targets] .+ precision,
                             params=get(gate, :params, []))
                else
                    add_gate!(circuit, gate[:gate], 
                             controls=[i], 
                             targets=gate[:targets] .+ precision,
                             params=get(gate, :params, []))
                end
            end
        end
    end
    
    # Apply inverse QFT to the first register
    iqft_circ = iqft(precision)
    for gate in iqft_circ.gates
        add_gate!(circuit, gate[:gate], 
                 controls=get(gate, :controls, []),
                 targets=gate[:targets],
                 params=get(gate, :params, []))
    end
    
    return circuit
end

"""
    dft_matrix(N::Int) -> Matrix{ComplexF64}

Generate the N×N Discrete Fourier Transform matrix:
  W[j,k] = exp(2π·im*(j-1)*(k-1)/N) / √N
"""
function dft_matrix(N::Int)
    ω = exp(2π * im / N)
    M = Matrix{ComplexF64}(undef, N, N)
    for j in 1:N, k in 1:N
        M[j, k] = ω^((j-1)*(k-1)) / sqrt(N)
    end
    return M
end

"""
    idft_matrix(N::Int) -> Matrix{ComplexF64}

Generate the inverse DFT matrix (iDFT):
  W⁻¹ = (W)†
"""
function idft_matrix(N::Int)
    conj.(transpose(dft_matrix(N)))
end

# Example usage:
# n_qubits = 3
# qft_circuit = qft(n_qubits)
# iqft_circuit = iqft(n_qubits)

# To verify, applying QFT and then iQFT should result in the identity.
# combined_circuit = qft_circuit + iqft_circuit 
# This should be equivalent to an empty circuit or identity operation.
end