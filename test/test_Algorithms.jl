using Test
using Heterotic.QSim
using Heterotic.Grovers
using Heterotic.QFT
using LinearAlgebra

@testset "Quantum Fourier Transform Tests" begin
    @testset "QFT followed by IQFT returns original state" begin
        # Test for different qubit sizes
        for n_qubits in 1:4
            # Create a circuit with QFT followed by IQFT
            circuit = QuantumCircuit(n_qubits)
            
            # Initialize to a non-trivial state
            for i in 1:n_qubits
                add_gate!(circuit, :h, targets=[i])
                add_gate!(circuit, :t, targets=[i])
            end
            
            # Store the state after initialization
            initial_circuit = deepcopy(circuit)
            initial_state = run_circuit(initial_circuit)
            
            # Apply QFT
            qft_circuit = qft(n_qubits)
            for gate in qft_circuit.gates
                add_gate!(circuit, gate[:gate], 
                          controls=get(gate, :controls, []),
                          targets=gate[:targets],
                          params=get(gate, :params, []))
            end
            
            # Apply IQFT
            iqft_circuit = iqft(n_qubits)
            for gate in iqft_circuit.gates
                add_gate!(circuit, gate[:gate], 
                          controls=get(gate, :controls, []),
                          targets=gate[:targets],
                          params=get(gate, :params, []))
            end
            
            # Run the full circuit
            final_state = run_circuit(circuit)
            
            # The final state should match the initial state
            initial_probs = get_measurement_probabilities(initial_state)
            final_probs = get_measurement_probabilities(final_state)
            
            for i in 1:2^n_qubits
                @test isapprox(initial_probs[i], final_probs[i], atol=1e-10)
            end
        end
    end
    
    @testset "QFT Unitary Matrix Properties" begin
        for n_qubits in 1:3
            # Get the QFT unitary matrix
            qft_circuit = qft(n_qubits)
            U = compute_unitary(qft_circuit)
            
            # QFT matrix should be unitary
            @test isapprox(U * U', I(2^n_qubits), atol=1e-10)
            
            # Test dimensions
            @test size(U) == (2^n_qubits, 2^n_qubits)
            
            # For n=1, QFT should be equivalent to Hadamard
            if n_qubits == 1
                @test isapprox(U, QSim.H, atol=1e-10)
            end
        end
    end
    
    @testset "QFT on Basis States" begin
        # Test QFT on |0⟩ for 2 qubits - should create equal superposition
        circuit = QuantumCircuit(2)
        
        # Apply QFT
        qft_circuit = qft(2)
        for gate in qft_circuit.gates
            add_gate!(circuit, gate[:gate], 
                      controls=get(gate, :controls, []),
                      targets=gate[:targets],
                      params=get(gate, :params, []))
        end
        
        # Run circuit
        state = run_circuit(circuit)
        probs = get_measurement_probabilities(state)
        
        # Should be in equal superposition
        for i in 1:4
            @test isapprox(probs[i], 0.25, atol=1e-10)
        end
    end
    
    
    @testset "Phase Estimation" begin
        # Test phase estimation with a known phase - using Z gate which has eigenvalues e^{i*π} = -1
        # For the |1⟩ state, Z|1⟩ = -1|1⟩, so phase is 0.5 (in units of 2π)
        
        # Create Z-rotation circuit (target circuit for phase estimation)
        z_circuit = QuantumCircuit(1)
        add_gate!(z_circuit, :z, targets=[1])
        
        # 3-qubit precision should give us a good approximation of 0.5
        precision = 3
        pe_circuit = phase_estimation(z_circuit, precision)
        
        # Prepare |1⟩ in the target register
        add_gate!(pe_circuit, :x, targets=[precision+1])
        
        # Run phase estimation
        state = run_circuit(pe_circuit)
        probs = get_measurement_probabilities(state)
        
        # Find the most probable state in the first register
        max_prob = 0
        max_state = 0
        
        for i in 0:(2^precision-1)
            # Extract only the phase register state (first 'precision' qubits)
            # For each possible phase register state, sum probabilities across all target register states
            phase_state_prob = sum(probs[i*(2^1) + j + 1] for j in 0:(2^1-1))
            
            if phase_state_prob > max_prob
                max_prob = phase_state_prob
                max_state = i
            end
        end
        
        # Expected phase 0.5 should be approximately encoded as 100 in binary (4 in decimal)
        # for 3 qubits precision
        @test max_state == 4
    end
end

@testset "Grover's Algorithm Tests" begin
    @testset "Oracle Construction" begin
        # Test oracle that marks state |1⟩
        circuit = QuantumCircuit(2)
        marked_states = [0]  # |00⟩ in zero-indexed notation (0 in decimal)
        add_gate!(circuit, :h, targets=[1])
        add_gate!(circuit, :h, targets=[2])
        create_oracle_circuit(circuit, marked_states)
    
        
        # Run on |00⟩ state
        state = run_circuit(circuit)

        # println(state)
        
        # State |00⟩ should have phase flipped
        if circuit.backend == :statevector
            # For statevector, check actual amplitudes
            @test isapprox(state.data[1], -0.5, atol=1e-10)
        else
            # For other backends, we can only check that probabilities haven't changed
            probs = get_measurement_probabilities(state)
            @test isapprox(probs[1], 1/4, atol=1e-10)
        end
        
        # Test oracle with multiple marked states
        circuit = QuantumCircuit(2)

        for i in 1:2
            add_gate!(circuit, :h, targets=[i])
        end
        marked_states = [0, 3] 
        
        # add_gate!(circuit, :h, targets=[1])
        # add_gate!(circuit, :h, targets=[2])
        # add_gate!(circuit, :x , targets= [1])
        # add_gate!(circuit,:x,targets=[2])
        # add_gate!(circuit, :h, targets=[2])
        # add_gate!(circuit, :cnot, controls = [1],targets=[2])
        # add_gate!(circuit, :h, targets=[2])
        # add_gate!(circuit, :x , targets= [1])
        # add_gate!(circuit,:x,targets=[2])
        # add_gate!(circuit, :h, targets=[1])
        # add_gate!(circuit, :h, targets=[2])

        create_diffusion_circuit(circuit)
        # Run circuit
        state = run_circuit(circuit)
        println("************")
        # println(state)
        display_state(state)
        probs = get_measurement_probabilities(state)
        
       
        @test isapprox(probs[1], probs[4], atol=1e-4)
        @test isapprox(probs[2], probs[3], atol=1e-4)
        @test isapprox(probs[1], probs[2], atol=1e-4)
    end
    
    
    @testset "Full Grover's Algorithm - Single Marked State" begin
        # Test full Grover's with one iteration on 2 qubits
        n_qubits = 2
        marked_states = [0]  # |00⟩
        iterations = 1
        
        circuit = create_grovers_circuit(n_qubits, marked_states, iterations)
        
        # Run circuit
        state = run_circuit(circuit)
        probs = get_measurement_probabilities(state)
        
        # After 1 iteration with 2 qubits, the marked state should have probability close to 1
        @test isapprox(probs[1], 1.0, atol=1e-1)
        
        # Test with 3 qubits and optimal number of iterations
        @testset "Additional Grover's Algorithm Tests" begin
            @testset "Hadamard Transformation" begin
                # Test that Hadamard gates create equal superposition
                circuit = QuantumCircuit(2)
                for i in 1:2
                    add_gate!(circuit, :h, targets=[i])
                end
                
                state = run_circuit(circuit)
                probs = get_measurement_probabilities(state)
                
                # Check for equal superposition
                for i in 1:4
                    @test isapprox(probs[i], 0.25, atol=1e-10)
                end
            end
            
            @testset "Oracle Single-Qubit Test" begin
                # Test a simple oracle on a single qubit
                circuit = QuantumCircuit(1)
                add_gate!(circuit, :h, targets=[1])
                
                # Mark state |0⟩
                create_oracle_circuit(circuit, [0])
                
                state = run_circuit(circuit)
                
                # Phase should be flipped for |0⟩
                if circuit.backend == :statevector
                    @test real(state.data[1]) < 0
                    @test real(state.data[2]) > 0
                end
                
                # Probabilities should still be 0.5 each
                probs = get_measurement_probabilities(state)
                @test isapprox(probs[1], 0.5, atol=1e-10)
                @test isapprox(probs[2], 0.5, atol=1e-10)
            end
            
            @testset "Diffusion Operator Test" begin
                # Test diffusion operator on |00⟩ state
                circuit = QuantumCircuit(2)
                
                # Apply diffusion to |00⟩ (should give -|00⟩ + 2/√N ∑|x⟩)
                create_diffusion_circuit(circuit)
                
                state = run_circuit(circuit)
                
                # In 2-qubit case with diffusion on |00⟩, all states should have equal probability
                probs = get_measurement_probabilities(state)
                for i in 1:4
                    @test isapprox(probs[i], 0.25, atol=1e-10)
                end
            end
            
            @testset "Grover's Iteration Test" begin
                # Test a single Grover iteration manually
                circuit = QuantumCircuit(2)
                
                # Initialize in superposition
                for i in 1:2
                    add_gate!(circuit, :h, targets=[i])
                end
                
                # Mark state |11⟩ (3 in zero-based indexing)
                create_oracle_circuit(circuit, [3])
                
                # Apply diffusion
                create_diffusion_circuit(circuit)
                
                # Run circuit
                state = run_circuit(circuit)
                probs = get_measurement_probabilities(state)
                
                # After one iteration, |11⟩ should have higher probability
                @test probs[4] > probs[1]
                @test probs[4] > probs[2]
                @test probs[4] > probs[3]
            end
            
            @testset "Small Complete Grover's" begin
                # Test a very small but complete Grover's algorithm
                n_qubits = 2
                marked_state = 3  # |11⟩
                
                circuit = QuantumCircuit(n_qubits)
                
                # Initial superposition
                for i in 1:n_qubits
                    add_gate!(circuit, :h, targets=[i])
                end
                
                # Oracle for |11⟩
                create_oracle_circuit(circuit, [marked_state])
                
                # Diffusion
                create_diffusion_circuit(circuit)
                
                # Run circuit
                state = run_circuit(circuit)
                probs = get_measurement_probabilities(state)
                
                # State |11⟩ should have highest probability
                @test probs[marked_state+1] == maximum(probs)
                @test probs[marked_state+1] > 0.5
            end
            
            @testset "Grover's With Custom Number of Iterations" begin
                # Test with different iteration counts
                n_qubits = 2
                marked_state = 2  # |10⟩
                
                # Try with 1 iteration (optimal for 2 qubits)
                circuit1 = create_grovers_circuit(n_qubits, [marked_state], 1)
                state1 = run_circuit(circuit1)
                probs1 = get_measurement_probabilities(state1)
                
                # Try with 2 iterations (over-rotation)
                circuit2 = create_grovers_circuit(n_qubits, [marked_state], 2)
                state2 = run_circuit(circuit2)
                probs2 = get_measurement_probabilities(state2)
                
                # First iteration should increase probability
                @test probs1[marked_state+1] > 0.5
                
                # Second iteration might decrease it due to over-rotation
                # Just test that both give different results
                @test !isapprox(probs1[marked_state+1], probs2[marked_state+1], atol=0.1)
            end
        end
    
    end
    

end