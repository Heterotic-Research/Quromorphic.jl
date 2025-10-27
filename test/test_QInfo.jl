using Test
using LinearAlgebra
using Quromorphic.QInfo

@testset "QInfo.jl Tests" begin
    
    @testset "Von Neumann Entropy" begin
        # Maximally mixed state (max entropy)
        ρ_max_mixed = ComplexF64[0.5 0.0; 0.0 0.5]
        @test von_neumann_entropy(ρ_max_mixed) ≈ 1.0 atol=1e-10
        
        # Pure state (zero entropy)
        ρ_pure = ComplexF64[1.0 0.0; 0.0 0.0]
        @test von_neumann_entropy(ρ_pure) ≈ 0.0 atol=1e-10
        
        # Mixed state
        ρ_mixed = ComplexF64[0.7 0.0; 0.0 0.3]
        S_expected = -0.7*log2(0.7) - 0.3*log2(0.3)
        @test von_neumann_entropy(ρ_mixed) ≈ S_expected atol=1e-10
    end
    
    @testset "Partial Trace" begin
        # Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        ψ_bell = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ρ_bell = ComplexF64.(ψ_bell * ψ_bell')
        
        # Reduced density matrices should be maximally mixed
        ρ_A = partial_trace(ρ_bell, 2, 2, 2)
        ρ_B = partial_trace(ρ_bell, 2, 2, 1)
        
        expected_reduced = ComplexF64[0.5 0.0; 0.0 0.5]
        @test ρ_A ≈ expected_reduced atol=1e-10
        @test ρ_B ≈ expected_reduced atol=1e-10
        
        # Product state |00⟩
        ψ_product = ComplexF64[1.0, 0.0, 0.0, 0.0]
        ρ_product = ψ_product * ψ_product'
        
        ρ_A_prod = partial_trace(ρ_product, 2, 2, 2)
        ρ_B_prod = partial_trace(ρ_product, 2, 2, 1)
        
        expected_pure = ComplexF64[1.0 0.0; 0.0 0.0]
        @test ρ_A_prod ≈ expected_pure atol=1e-10
        @test ρ_B_prod ≈ expected_pure atol=1e-10
    end
    
    @testset "Entanglement Entropy" begin
        # Bell state - maximally entangled
        ψ_bell = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ρ_bell = ComplexF64.(ψ_bell * ψ_bell')
        @test entanglement_entropy(ρ_bell, 2, 2) ≈ 1.0 atol=1e-10
        
        # Product state - not entangled
        ψ_product = ComplexF64[1.0, 0.0, 0.0, 0.0]
        ρ_product = ψ_product * ψ_product'
        @test entanglement_entropy(ρ_product, 2, 2) ≈ 0.0 atol=1e-10
    end
    
    @testset "Quantum Mutual Information" begin
        # Bell state - maximum correlations
        ψ_bell = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ρ_bell = ComplexF64.(ψ_bell * ψ_bell')
        I_AB = quantum_mutual_information(ρ_bell, 2, 2)
        @test I_AB ≈ 2.0 atol=1e-10  # I(A:B) = S(A) + S(B) - S(AB) = 1 + 1 - 0 = 2
        
        # Product state - no correlations
        ρ_prod = ComplexF64[0.5 0.0 0.0 0.0;
                            0.0 0.5 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0]
        @test quantum_mutual_information(ρ_prod, 2, 2) ≈ 0.0 atol=1e-10
    end
    
    @testset "Conditional Quantum Entropy" begin
        # Bell state
        ψ_bell = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ρ_bell = ComplexF64.(ψ_bell * ψ_bell')
        S_cond = conditional_quantum_entropy(ρ_bell, 2, 2)
        @test S_cond ≈ -1.0 atol=1e-10  # S(A|B) = S(AB) - S(B) = 0 - 1 = -1
        
        # Product state
        ρ_prod = ComplexF64[1.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0]
        @test conditional_quantum_entropy(ρ_prod, 2, 2) ≈ 0.0 atol=1e-10
    end
    
    @testset "Coherent Information" begin
        # Bell state
        ψ_bell = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ρ_bell = ComplexF64.(ψ_bell * ψ_bell')
        I_c = coherent_information(ρ_bell, 2, 2)
        @test I_c ≈ 1.0 atol=1e-10  # I_c = S(B) - S(AB) = 1 - 0 = 1
    end
    
    @testset "Relative Entropy" begin
        ρ = ComplexF64[0.7 0.0; 0.0 0.3]
        σ = ComplexF64[0.5 0.0; 0.0 0.5]
        
        # D(ρ||σ) should be non-negative
        D = relative_entropy(ρ, σ)
        @test D >= 0.0
        
        # D(ρ||ρ) = 0
        @test relative_entropy(ρ, ρ) ≈ 0.0 atol=1e-10
        
        # Analytical value for diagonal states
        D_expected = 0.7*log2(0.7/0.5) + 0.3*log2(0.3/0.5)
        @test D ≈ D_expected atol=1e-10
    end
    
    @testset "Holevo Information" begin
        # Two pure orthogonal states
        ρ1 = ComplexF64[1.0 0.0; 0.0 0.0]
        ρ2 = ComplexF64[0.0 0.0; 0.0 1.0]
        states = [ρ1, ρ2]
        probs = [0.5, 0.5]
        
        χ = holevo_information(states, probs)
        @test χ ≈ 1.0 atol=1e-10  # Maximum for this case
        
        # Same state twice - zero information
        states_same = [ρ1, ρ1]
        @test holevo_information(states_same, probs) ≈ 0.0 atol=1e-10
    end
    
    @testset "Rényi Entropy" begin
        ρ = ComplexF64[0.5 0.0; 0.0 0.5]
        
        # For maximally mixed state, Rényi entropy = 1 for all α
        @test renyi_entropy(ρ, 0.5) ≈ 1.0 atol=1e-10
        @test renyi_entropy(ρ, 2.0) ≈ 1.0 atol=1e-10
        @test renyi_entropy(ρ, 10.0) ≈ 1.0 atol=1e-10
        
        # For pure state
        ρ_pure = ComplexF64[1.0 0.0; 0.0 0.0]
        @test renyi_entropy(ρ_pure, 2.0) ≈ 0.0 atol=1e-10
        
        # Test α=1 assertion
        @test_throws AssertionError renyi_entropy(ρ, 1.0)
    end
    
    @testset "Quantum Relative Rényi Entropy" begin
        ρ = ComplexF64[0.7 0.0; 0.0 0.3]
        σ = ComplexF64[0.5 0.0; 0.0 0.5]
        
        # Should be non-negative
        D2 = quantum_relative_renyi_entropy(ρ, σ, 2.0)
        @test D2 >= 0.0
        
        # D(ρ||ρ) = 0
        @test quantum_relative_renyi_entropy(ρ, ρ, 2.0) ≈ 0.0 atol=1e-10
        
        # Test α=1 assertion
        @test_throws AssertionError quantum_relative_renyi_entropy(ρ, σ, 1.0)
    end
    
    @testset "Quantum Fisher Information" begin
        # Example: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        θ = π/4
        ψ = ComplexF64[cos(θ/2), sin(θ/2)]
        dψ_dθ = ComplexF64[-sin(θ/2)/2, cos(θ/2)/2]
        
        F_Q = quantum_fisher_information(ψ, dψ_dθ)
        @test F_Q >= 0.0  # Fisher information is always non-negative
        @test F_Q ≈ 1.0 atol=1e-10  # For this parametrization
        
        # For |0⟩ state with no dependence on θ
        ψ_0 = ComplexF64[1.0, 0.0]
        dψ_0 = ComplexF64[0.0, 0.0]
        @test quantum_fisher_information(ψ_0, dψ_0) ≈ 0.0 atol=1e-10
    end
    
    @testset "l1-norm Coherence" begin
        # Pure state with coherence
        ρ_coherent = ComplexF64[0.5 0.5; 0.5 0.5]
        @test l1_norm_coherence(ρ_coherent) ≈ 1.0 atol=1e-10
        
        # Diagonal state (no coherence)
        ρ_diagonal = ComplexF64[0.7 0.0; 0.0 0.3]
        @test l1_norm_coherence(ρ_diagonal) ≈ 0.0 atol=1e-10
        
        # Partially coherent state
        ρ_partial = ComplexF64[0.6 0.2; 0.2 0.4]
        @test l1_norm_coherence(ρ_partial) ≈ 0.4 atol=1e-10
    end
    
    @testset "Fidelity" begin
        # Fidelity with itself is 1
        ρ = ComplexF64[0.7 0.0; 0.0 0.3]
        @test fidelity(ρ, ρ) ≈ 1.0 atol=1e-10
        
        # Fidelity is bounded [0, 1]
        σ = ComplexF64[0.3 0.0; 0.0 0.7]
        F = fidelity(ρ, σ)
        @test 0.0 <= F <= 1.0
        @test F ≈ 0.84 atol=0.01  # ≈ 2*sqrt(0.7*0.3*0.3*0.7) + 0.7*0.3 + 0.3*0.7
        
        # Orthogonal pure states have zero fidelity
        ρ1 = ComplexF64[1.0 0.0; 0.0 0.0]
        ρ2 = ComplexF64[0.0 0.0; 0.0 1.0]
        @test fidelity(ρ1, ρ2) ≈ 0.0 atol=1e-10
        
        # Same pure states have fidelity 1
        @test fidelity(ρ1, ρ1) ≈ 1.0 atol=1e-10
    end
    
    @testset "Trace Distance" begin
        # Trace distance with itself is 0
        ρ = ComplexF64[0.7 0.0; 0.0 0.3]
        @test trace_distance(ρ, ρ) ≈ 0.0 atol=1e-10
        
        # Trace distance is bounded [0, 1]
        σ = ComplexF64[0.3 0.0; 0.0 0.7]
        T = trace_distance(ρ, σ)
        @test 0.0 <= T <= 1.0
        @test T ≈ 0.4 atol=1e-10  # For diagonal states: 0.5*|0.7-0.3 + 0.3-0.7|
        
        # Orthogonal pure states have maximal distance
        ρ1 = ComplexF64[1.0 0.0; 0.0 0.0]
        ρ2 = ComplexF64[0.0 0.0; 0.0 1.0]
        @test trace_distance(ρ1, ρ2) ≈ 1.0 atol=1e-10
    end
    
    @testset "Max-Entropy" begin
        # Rank-2 state
        ρ = ComplexF64[0.5 0.0; 0.0 0.5]
        @test max_entropy(ρ) ≈ 1.0 atol=1e-10  # log2(2) = 1
        
        # Rank-1 state (pure)
        ρ_pure = ComplexF64[1.0 0.0; 0.0 0.0]
        @test max_entropy(ρ_pure) ≈ 0.0 atol=1e-10  # log2(1) = 0
        
        # Rank-3 state
        ρ_3 = ComplexF64[0.4 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3]
        @test max_entropy(ρ_3) ≈ log2(3) atol=1e-10
    end
    
    @testset "Min-Entropy" begin
        # Maximally mixed state
        ρ = ComplexF64[0.5 0.0; 0.0 0.5]
        @test min_entropy(ρ) ≈ -log2(0.5) atol=1e-10  # = 1
        
        # Pure state
        ρ_pure = ComplexF64[1.0 0.0; 0.0 0.0]
        @test min_entropy(ρ_pure) ≈ -log2(1.0) atol=1e-10  # = 0
        
        # Non-uniform mixed state
        ρ_mixed = ComplexF64[0.8 0.0; 0.0 0.2]
        @test min_entropy(ρ_mixed) ≈ -log2(0.8) atol=1e-10
    end
    
    @testset "Consistency Checks" begin
        # Von Neumann entropy bounds
        ρ = ComplexF64[0.7 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.1]
        S_vN = von_neumann_entropy(ρ)
        S_max = max_entropy(ρ)
        S_min = min_entropy(ρ)
        
        # S_min ≤ S_vN ≤ S_max
        @test S_min <= S_vN + 1e-10
        @test S_vN <= S_max + 1e-10
        
        # For pure states, all entropies should be zero
        ψ = ComplexF64[1.0, 0.0, 0.0]
        ρ_pure = ψ * ψ'
        @test von_neumann_entropy(ρ_pure) ≈ 0.0 atol=1e-10
        @test max_entropy(ρ_pure) ≈ 0.0 atol=1e-10
        @test min_entropy(ρ_pure) ≈ 0.0 atol=1e-10
    end
    

    

    @testset "Bipartite State Tests" begin
        # Werner state: W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4
        # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        p = 0.7
        ψ_singlet = [0.0, 1.0, -1.0, 0.0] / sqrt(2)
        ρ_singlet = ComplexF64.(ψ_singlet * ψ_singlet')
        I4 = ComplexF64.(Matrix(I, 4, 4))
        ρ_werner = p * ρ_singlet + (1-p) * I4 / 4
        
        # Test various measures
        S_AB = von_neumann_entropy(ρ_werner)
        I_AB = quantum_mutual_information(ρ_werner, 2, 2)
        S_cond = conditional_quantum_entropy(ρ_werner, 2, 2)
        
        @test S_AB >= 0.0
        @test I_AB >= 0.0
        # For Werner states, I(A:B) = 2 - S(AB)
        @test abs(I_AB + S_AB - 2.0) < 1e-10  # Consistency check
    end
    
    @testset "GHZ State Tests" begin
        # GHZ state: (|000⟩ + |111⟩)/√2 (3 qubits)
        ψ_ghz = zeros(ComplexF64, 8)
        ψ_ghz[1] = 1/sqrt(2)   # |000⟩
        ψ_ghz[8] = 1/sqrt(2)   # |111⟩
        ρ_ghz = ψ_ghz * ψ_ghz'
        
        # Trace out one qubit (2x4 system)
        ρ_reduced = partial_trace(ρ_ghz, 2, 4, 1)
        S_reduced = von_neumann_entropy(ρ_reduced)
        
        # GHZ state has interesting entanglement structure
        @test S_reduced > 0.0  # Should have entropy after tracing out
    end
    
    @testset "Edge Cases" begin
        # Very small off-diagonal elements
        ρ = ComplexF64[1.0-1e-12 1e-13; 1e-13 1e-12]
        @test von_neumann_entropy(ρ) >= 0.0
        
        # Nearly maximally mixed
        ρ_nearly_mixed = ComplexF64[0.5+1e-10 0.0; 0.0 0.5-1e-10]
        @test abs(von_neumann_entropy(ρ_nearly_mixed) - 1.0) < 1e-8
        
        # Single qubit states
        ρ_single = ComplexF64[0.6 0.2; 0.2 0.4]
        @test 0.0 <= von_neumann_entropy(ρ_single) <= 1.0
    end
    
    @testset "Complex Coherence Tests" begin
        # State with complex off-diagonal elements
        ρ_complex = ComplexF64[0.5 0.3im; -0.3im 0.5]
        coh = l1_norm_coherence(ρ_complex)
        @test coh ≈ 0.6 atol=1e-10  # |0.3i| + |-0.3i| = 0.3 + 0.3
        
        # Ensure hermiticity is preserved
        @test ρ_complex ≈ ρ_complex' atol=1e-10
    end
    




    @testset "Numerical Stability" begin
        # Test with small eigenvalues
        λ = ComplexF64.([0.999, 1e-10, 1e-11])
        λ = λ / sum(λ)
        ρ = diagm(λ)
        
        @test von_neumann_entropy(ρ) >= 0.0
        @test !isnan(von_neumann_entropy(ρ))
        @test !isinf(von_neumann_entropy(ρ))
        
        # Test renyi entropy stability
        @test !isnan(renyi_entropy(ρ, 2.0))
        @test !isinf(renyi_entropy(ρ, 2.0))
    end
end
