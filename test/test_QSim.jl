using Test

using Quromorphic.QSim
using LinearAlgebra


@testset "QSim Full Test Suite" begin
    
    @testset "State Vector Creation and Basic Operations" begin
        # Test statevector creation
        @testset "State Vector Creation" begin
            s1 = statevector(2, 0)
            @test length(s1) == 4
            @test s1[1] ≈ 1.0 + 0.0im
            @test all(s1[2:end] .≈ 0.0)
            
            s2 = statevector(3, 5)
            @test length(s2) == 8
            @test s2[6] ≈ 1.0 + 0.0im
            @test sum(abs2.(s2)) ≈ 1.0
        end
        
        # Test nb function
        @testset "Number of Qubits Function" begin
            s = statevector(3, 0)
            @test nb(s) == 3
            
            s2 = statevector(5, 0)
            @test nb(s2) == 5
        end
        
        # Test mp function
        @testset "Measurement Probabilities" begin
            s = statevector(2, 1)
            probs = mp(s)
            @test length(probs) == 4
            @test probs[2] ≈ 1.0
            @test sum(probs) ≈ 1.0
        end
    end
    
    @testset "Density Matrix Operations" begin
        @testset "Density Matrix Creation" begin
            rho = density_matrix(2, 0)
            @test size(rho) == (4, 4)
            @test rho[1,1] ≈ 1.0
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)
            
            # Test conversion from state vector
            s = statevector(2, 1)
            rho2 = statevector_to_density_matrix(s)
            @test size(rho2) == (4, 4)
            @test tr(rho2) ≈ 1.0
            @test ishermitian(rho2)
        end
        
        @testset "Density Matrix nb and mp" begin
            rho = density_matrix(3, 0)
            @test nb(rho) == 3
            
            probs = mp(rho)
            @test length(probs) == 8
            @test probs[1] ≈ 1.0
            @test sum(probs) ≈ 1.0
        end
    end
    
    @testset "Single Qubit Gates - State Vector" begin
        @testset "Hadamard Gate" begin
            s = statevector(1, 0)
            h!(s, 1)
            @test abs(s[1]) ≈ 1/√2
            @test abs(s[2]) ≈ 1/√2
            @test sum(abs2.(s)) ≈ 1.0
            
            # Test on multi-qubit state
            s2 = statevector(2, 0)
            h!(s2, 1)
            @test abs(s2[1]) ≈ 1/√2
            @test abs(s2[2]) ≈ 1/√2
        end
        
        @testset "Pauli Gates" begin
            # X gate
            s = statevector(1, 0)
            x!(s, 1)
            @test s[1] ≈ 0.0
            @test s[2] ≈ 1.0
            
            # Y gate
            s = statevector(1, 0)
            y!(s, 1)
            @test s[1] ≈ 0.0
            @test abs(s[2]) ≈ 1.0
            
            # Z gate
            s = statevector(1, 1)
            z!(s, 1)
            @test s[1] ≈ 0.0
            @test s[2] ≈ -1.0
        end
        
        @testset "Rotation Gates" begin
            # RX gate
            s = statevector(1, 0)
            rx!(s, 1, π)
            @test abs(s[1]) < 1e-10
            @test abs(s[2]) ≈ 1.0
            
            # RY gate
            s = statevector(1, 0)
            ry!(s, 1, π/2)
            @test abs(s[1]) ≈ 1/√2
            @test abs(s[2]) ≈ 1/√2
            
            # RZ gate
            s = statevector(1, 0)
            rz!(s, 1, π/2)
            @test sum(abs2.(s)) ≈ 1.0
        end
    end
    
    @testset "Single Qubit Gates - Density Matrix" begin
        @testset "Hadamard Gate on Density Matrix" begin
            rho = density_matrix(1, 0)
            h!(rho, 1)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)
            @test real(rho[1,1]) ≈ 0.5
            @test real(rho[2,2]) ≈ 0.5
        end
        
        @testset "Pauli Gates on Density Matrix" begin
            # X gate
            rho = density_matrix(1, 0)
            x!(rho, 1)
            @test real(rho[1,1]) ≈ 0.0
            @test real(rho[2,2]) ≈ 1.0
            @test tr(rho) ≈ 1.0
            
            # Y gate
            rho = density_matrix(1, 0)
            y!(rho, 1)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)
            
            # Z gate
            rho = density_matrix(1, 1)
            z!(rho, 1)
            @test real(rho[2,2]) ≈ 1.0
            @test tr(rho) ≈ 1.0
        end
        
        @testset "Rotation Gates on Density Matrix" begin
            rho = density_matrix(1, 0)
            rx!(rho, 1, π/4)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)
            
            rho2 = density_matrix(1, 0)
            ry!(rho2, 1, π/3)
            @test tr(rho2) ≈ 1.0
            @test ishermitian(rho2)
            
            rho3 = density_matrix(1, 0)
            rz!(rho3, 1, π/6)
            @test tr(rho3) ≈ 1.0
            @test ishermitian(rho3)
        end
    end
    
    @testset "Two Qubit Gates" begin
        @testset "CNOT Gate - State Vector" begin
            # |00⟩ → |00⟩
            s = statevector(2, 0)
            cnot!(s, 2, 1)
            @test s[1] ≈ 1.0
            @test all(s[2:end] .≈ 0.0)
            
            # |10⟩ → |11⟩
            s = statevector(2, 2)
            cnot!(s, 2, 1)
            @test s[3] ≈ 0.0
            @test s[4] ≈ 1.0
            
            # Test reverse control
            s = statevector(2, 1) # |01⟩
            cnot!(s, 1, 2)
            @test s[2] ≈ 0.0
            @test s[4] ≈ 1.0
        end
        
        @testset "CNOT Gate - Density Matrix" begin
            rho = density_matrix(2, 0)
            cnot!(rho, 1, 2)
            @test real(rho[1,1]) ≈ 1.0
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)
            
            rho2 = density_matrix(2, 2) # |10> 
            cnot!(rho2, 2, 1)
            @test real(rho2[4,4]) ≈ 1.0
            @test tr(rho2) ≈ 1.0
        end
        
        @testset "SWAP Gate" begin
            # State vector
            s = statevector(2, 1)  # |01⟩
            swap!(s, 1, 2)
            @test s[2] ≈ 0.0
            @test s[3] ≈ 1.0  # Should be |10⟩
            
            # Density matrix
            rho = density_matrix(2, 1)
            swap!(rho, 1, 2)
            @test real(rho[3,3]) ≈ 1.0
            @test tr(rho) ≈ 1.0
            
            # Test swap with same qubit (should be no-op)
            s2 = statevector(2, 1)
            s_original = copy(s2)
            swap!(s2, 1, 1)
            @test s2 ≈ s_original
        end
        
        @testset "Controlled Rotation Gates" begin
            # CRX
            s = statevector(2, 2)  # |10⟩
            crx!(s, 1, 2, π)
            @test sum(abs2.(s)) ≈ 1.0
            
            # CRY
            s = statevector(2, 2)
            cry!(s, 1, 2, π/2)
            @test sum(abs2.(s)) ≈ 1.0
            
            # CRZ
            s = statevector(2, 2)
            crz!(s, 1, 2, π/4)
            @test sum(abs2.(s)) ≈ 1.0
            
            # Test on density matrices
            rho = density_matrix(2, 2)
            crx!(rho, 1, 2, π/4)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)

            # Test CRY on density matrices
            rho = density_matrix(2, 2)
            cry!(rho, 1, 2, π/4)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)

            # Test CRZ on density matrices
            rho = density_matrix(2, 2)
            crz!(rho, 1, 2, π/4)
            @test tr(rho) ≈ 1.0
            @test ishermitian(rho)

        end
    end
    
    @testset "Measurement Functions" begin
        @testset "Z-basis Measurements" begin
            # State vector measurements
            s = statevector(2, 0)  # |00⟩
            p0, p1 = measure_z(s, 1)
            @test p0 ≈ 1.0
            @test p1 ≈ 0.0
            
            s = statevector(2, 2)  # |10⟩
            p0, p1 = measure_z(s, 2)
            @test p0 ≈ 0.0
            @test p1 ≈ 1.0
            
            # Superposition state
            s = statevector(1, 0)
            h!(s, 1)
            p0, p1 = measure_z(s, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
            
            # Density matrix measurements
            rho = density_matrix(1, 0)
            h!(rho, 1)
            p0, p1 = measure_z(rho, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
        end
        
        @testset "X-basis Measurements" begin
            s = statevector(1, 0)
            p0, p1 = measure_x(s, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
            
            rho = density_matrix(1, 0)
            p0, p1 = measure_x(rho, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
        end
        
        @testset "Y-basis Measurements" begin
            s = statevector(1, 0)
            p0, p1 = measure_y(s, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
            
            rho = density_matrix(1, 0)
            p0, p1 = measure_y(rho, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
        end
    end
    
    @testset "Utility Functions" begin
        @testset "Print State Function" begin
            s = statevector(2, 0)
            # Test that prstate doesn't error (output testing is complex)
            @test_nowarn prstate(s)
            
            rho = density_matrix(2, 0)
            @test_nowarn prstate(rho)
        end
        
        @testset "Identity Function" begin
            I1 = id(1)
            @test size(I1) == (2, 2)
            @test I1 ≈ [1 0; 0 1]
            
            I2 = id(2)
            @test size(I2) == (4, 4)
            @test I2[1,1] ≈ 1.0
            @test I2[4,4] ≈ 1.0
            
            I0 = id(0)
            @test size(I0) == (1, 1)
            @test I0[1,1] ≈ 1.0
        end
    end
    

    
    @testset "Error Handling" begin
        @testset "Dimension Mismatch Errors" begin
            s = statevector(2, 0)
        
            
            # Qubit index out of range
            @test_throws ErrorException h!(s, 3)
            @test_throws ErrorException x!(s, 6)
            @test_throws ErrorException measure_z(s, 5)
            
            # CNOT with same control and target
            @test_throws ErrorException cnot!(s, 1, 1)
            
            # Out of range qubits for two-qubit gates
            @test_throws ErrorException cnot!(s, 1, 3)
            @test_throws ErrorException swap!(s, 1, 7)
        end
        
        @testset "Density Matrix Error Handling" begin
            rho = density_matrix(2, 0)
            
            # Qubit index out of range
            @test_throws ErrorException h!(rho, 3)
            @test_throws ErrorException cnot!(rho, 1, 3)
            @test_throws ErrorException measure_z(rho, 3)
            
            # Same control and target
            @test_throws ErrorException cnot!(rho, 1, 1)
        end
    end
    
    @testset "Gate Caching" begin
        @testset "Rotation Gate Caching" begin
            # Test that repeated calls use cached gates
            s1 = statevector(1, 0)
            s2 = statevector(1, 0)
            
            rx!(s1, 1, π/4)
            rx!(s2, 1, π/4)
            
            @test s1 ≈ s2
            
            # Test cache for different angles
            s3 = statevector(1, 0)
            rx!(s3, 1, π/3)
            @test !(s1 ≈ s3)
        end
    end
    
    @testset "Complex Multi-Qubit Circuits" begin
        @testset "Bell State Creation" begin
            # Create |Φ+⟩ = (|00⟩ + |11⟩)/√2
            s = statevector(2, 0)
            h!(s, 1)
            cnot!(s, 1, 2)
            
            @test abs(s[1]) ≈ 1/√2
            @test abs(s[4]) ≈ 1/√2
            @test sum(abs2.(s)) ≈ 1.0
            
            # Measure correlations
            p0, p1 = measure_z(s, 1)
            @test p0 ≈ 0.5
            @test p1 ≈ 0.5
        end
        
        @testset "GHZ State Creation" begin
            # Create |GHZ⟩ = (|000⟩ + |111⟩)/√2
            s = statevector(3, 0)
            h!(s, 1)
            cnot!(s, 1, 2)
            cnot!(s, 1, 3)
            
            @test abs(s[1]) ≈ 1/√2
            @test abs(s[8]) ≈ 1/√2
            @test sum(abs2.(s)) ≈ 1.0
        end
    end
    
    @testset "Numerical Precision" begin
        @testset "Very Small Amplitudes" begin
            s = statevector(2, 0)
            h!(s, 1)
            h!(s, 2)
            
            # Apply many operations and check normalization
            for i in 1:10
                rx!(s, 1, 0.1)
                ry!(s, 2, 0.1)
                cnot!(s, 1, 2)
            end
            
            @test abs(sum(abs2.(s)) - 1.0) < 1e-10
        end
    end
end
