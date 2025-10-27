module QInfo

using LinearAlgebra

export von_neumann_entropy, quantum_mutual_information, conditional_quantum_entropy,
       coherent_information, relative_entropy, holevo_information, entanglement_entropy,
       renyi_entropy, quantum_relative_renyi_entropy, quantum_fisher_information, 
       l1_norm_coherence, fidelity, trace_distance, max_entropy, min_entropy, partial_trace

"""
    matrix_log2(ρ::Matrix{ComplexF64})

Compute the base-2 logarithm of a density matrix.

# Arguments
- `ρ::Matrix{ComplexF64}`: Hermitian density matrix

# Returns
- `Matrix{ComplexF64}`: Matrix logarithm in base 2

# Details
Uses eigendecomposition and sets a cutoff of 1e-10 for small eigenvalues to avoid log(0).
"""
function matrix_log2(ρ::Matrix{ComplexF64})
    eigvals, eigvecs = eigen(Hermitian(ρ))
    eigvals = max.(eigvals, 1e-10)
    return eigvecs * Diagonal(log2.(eigvals)) * eigvecs'
end

"""
    von_neumann_entropy(ρ::Matrix{ComplexF64})

Calculate the von Neumann entropy of a quantum state.

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix of the quantum state

# Returns
- `Float64`: Von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ)

# Examples

ρ = [0.5 0.0; 0.0 0.5]  # Maximally mixed state
S = von_neumann_entropy(ρ)  # Returns 1.0

"""
function von_neumann_entropy(ρ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    return -real(tr(ρ * matrix_log2(ρ)))
end

"""
    quantum_mutual_information(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)

Calculate the quantum mutual information between subsystems A and B.

# Arguments
- `ρ_AB::Matrix{ComplexF64}`: Joint density matrix of the bipartite system
- `dim_A::Int`: Dimension of subsystem A
- `dim_B::Int`: Dimension of subsystem B

# Returns
- `Float64`: Quantum mutual information I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

# Details
Measures the total correlations (classical and quantum) between subsystems A and B.
"""
function quantum_mutual_information(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)
    ρ_AB = (ρ_AB + ρ_AB') / 2
    ρ_A = partial_trace(ρ_AB, dim_A, dim_B, 2)
    ρ_B = partial_trace(ρ_AB, dim_A, dim_B, 1)
    return von_neumann_entropy(ρ_A) + von_neumann_entropy(ρ_B) - von_neumann_entropy(ρ_AB)
end

"""
    conditional_quantum_entropy(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)

Calculate the conditional quantum entropy S(A|B).

# Arguments
- `ρ_AB::Matrix{ComplexF64}`: Joint density matrix of the bipartite system
- `dim_A::Int`: Dimension of subsystem A
- `dim_B::Int`: Dimension of subsystem B

# Returns
- `Float64`: Conditional entropy S(A|B) = S(ρ_AB) - S(ρ_B)

# Details
Can be negative for entangled states, unlike classical conditional entropy.
"""
function conditional_quantum_entropy(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)
    ρ_AB = (ρ_AB + ρ_AB') / 2
    ρ_B = partial_trace(ρ_AB, dim_A, dim_B, 1)
    return von_neumann_entropy(ρ_AB) - von_neumann_entropy(ρ_B)
end

"""
    coherent_information(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)

Calculate the coherent information from A to B.

# Arguments
- `ρ_AB::Matrix{ComplexF64}`: Joint density matrix of the bipartite system
- `dim_A::Int`: Dimension of subsystem A
- `dim_B::Int`: Dimension of subsystem B

# Returns
- `Float64`: Coherent information I_c(A⟩B) = S(ρ_B) - S(ρ_AB)

# Details
Related to the quantum channel capacity and entanglement transmission.
"""
function coherent_information(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)
    ρ_AB = (ρ_AB + ρ_AB') / 2
    ρ_B = partial_trace(ρ_AB, dim_A, dim_B, 1)
    return von_neumann_entropy(ρ_B) - von_neumann_entropy(ρ_AB)
end

"""
    relative_entropy(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})

Calculate the quantum relative entropy (Kullback-Leibler divergence).

# Arguments
- `ρ::Matrix{ComplexF64}`: First density matrix
- `σ::Matrix{ComplexF64}`: Second density matrix

# Returns
- `Float64`: Relative entropy S(ρ || σ) = Tr(ρ log ρ) - Tr(ρ log σ)

# Details
Measures distinguishability between quantum states. Always non-negative and zero iff ρ = σ.
"""
function relative_entropy(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    σ = (σ + σ') / 2
    return real(tr(ρ * matrix_log2(ρ)) - tr(ρ * matrix_log2(σ)))
end

"""
    holevo_information(states::Vector{Matrix{ComplexF64}}, probs::Vector{Float64})

Calculate the Holevo bound (accessible information).

# Arguments
- `states::Vector{Matrix{ComplexF64}}`: Ensemble of quantum states
- `probs::Vector{Float64}`: Probability distribution over states (must sum to 1)

# Returns
- `Float64`: Holevo information χ = S(∑ pₓ ρₓ) - ∑ pₓ S(ρₓ)

# Details
Upper bound on the classical information that can be extracted from the ensemble.
"""
function holevo_information(states::Vector{Matrix{ComplexF64}}, probs::Vector{Float64})
    @assert abs(sum(probs) - 1.0) < 1e-10 "Probabilities must sum to 1"
    ρ_avg = sum(p * ρ for (p, ρ) in zip(probs, states))
    S_individual = sum(p * von_neumann_entropy(ρ) for (p, ρ) in zip(probs, states))
    return von_neumann_entropy(ρ_avg) - S_individual
end

"""
    entanglement_entropy(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)

Calculate the entanglement entropy of a bipartite state.

# Arguments
- `ρ_AB::Matrix{ComplexF64}`: Joint density matrix of the bipartite system
- `dim_A::Int`: Dimension of subsystem A
- `dim_B::Int`: Dimension of subsystem B

# Returns
- `Float64`: Entanglement entropy S(ρ_A) = S(ρ_B) for pure states

# Details
For pure bipartite states, quantifies the entanglement between subsystems.
"""
function entanglement_entropy(ρ_AB::Matrix{ComplexF64}, dim_A::Int, dim_B::Int)
    ρ_AB = (ρ_AB + ρ_AB') / 2
    ρ_A = partial_trace(ρ_AB, dim_A, dim_B, 2)
    return von_neumann_entropy(ρ_A)
end

"""
    renyi_entropy(ρ::Matrix{ComplexF64}, α::Float64)

Calculate the Rényi entropy of order α.

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix
- `α::Float64`: Order parameter (α ≠ 1)

# Returns
- `Float64`: Rényi entropy Sα(ρ) = (1/(1-α)) log₂ Tr(ρᵅ)

# Details
Generalizes von Neumann entropy. For α → 1, converges to von Neumann entropy.
"""
function renyi_entropy(ρ::Matrix{ComplexF64}, α::Float64)
    @assert α != 1.0 "Rényi entropy undefined for α=1; use von Neumann entropy"
    ρ = (ρ + ρ') / 2
    ρ_α = Hermitian(ρ)^α
    return (1/(1-α)) * log2(real(tr(ρ_α)))
end

"""
    quantum_relative_renyi_entropy(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64}, α::Float64)

Calculate the quantum relative Rényi entropy.

# Arguments
- `ρ::Matrix{ComplexF64}`: First density matrix
- `σ::Matrix{ComplexF64}`: Second density matrix
- `α::Float64`: Order parameter (α ≠ 1)

# Returns
- `Float64`: Quantum relative Rényi entropy Dα(ρ || σ) = (1/(α-1)) log₂ Tr(ρᵅ σ^(1-α))

# Details
Generalizes quantum relative entropy for distinguishability measures.
"""
function quantum_relative_renyi_entropy(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64}, α::Float64)
    @assert α != 1.0 "Relative Rényi entropy undefined for α=1"
    ρ = (ρ + ρ') / 2
    σ = (σ + σ') / 2
    return (1/(α-1)) * log2(real(tr(Hermitian(ρ)^α * Hermitian(σ)^(1-α))))
end

"""
    quantum_fisher_information(ψ::Vector{ComplexF64}, dψ_dθ::Vector{ComplexF64})

Calculate the quantum Fisher information for a pure state.

# Arguments
- `ψ::Vector{ComplexF64}`: Pure state vector (normalized)
- `dψ_dθ::Vector{ComplexF64}`: Derivative of state with respect to parameter θ

# Returns
- `Float64`: Quantum Fisher information F_Q = 4(⟨dψ|dψ⟩ - |⟨ψ|dψ⟩|²)

# Details
Fundamental limit for parameter estimation in quantum metrology (Cramér-Rao bound).
"""
function quantum_fisher_information(ψ::Vector{ComplexF64}, dψ_dθ::Vector{ComplexF64})
    return 4 * real(dot(dψ_dθ, dψ_dθ) - abs2(dot(ψ, dψ_dθ)))
end

"""
    l1_norm_coherence(ρ::Matrix{ComplexF64})

Calculate the l1-norm of coherence.

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix

# Returns
- `Float64`: l1-coherence C_l1(ρ) = ∑_{i≠j} |ρᵢⱼ|

# Details
Quantifies quantum coherence as the sum of absolute values of off-diagonal elements.
"""
function l1_norm_coherence(ρ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    n = size(ρ, 1)
    coh = 0.0
    for i in 1:n, j in 1:n
        if i != j
            coh += abs(ρ[i,j])
        end
    end
    return coh
end

"""
    fidelity(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})

Calculate the fidelity between two quantum states.

# Arguments
- `ρ::Matrix{ComplexF64}`: First density matrix
- `σ::Matrix{ComplexF64}`: Second density matrix

# Returns
- `Float64`: Fidelity F(ρ, σ) = [Tr√(√ρ σ √ρ)]²

# Details
Measures closeness of quantum states. F = 1 for identical states, F = 0 for orthogonal states.
"""
function fidelity(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    σ = (σ + σ') / 2
    sqrt_ρ = sqrt(Hermitian(ρ))
    return real(tr(sqrt(Hermitian(sqrt_ρ * σ * sqrt_ρ))))^2
end

"""
    trace_distance(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})

Calculate the trace distance between two quantum states.

# Arguments
- `ρ::Matrix{ComplexF64}`: First density matrix
- `σ::Matrix{ComplexF64}`: Second density matrix

# Returns
- `Float64`: Trace distance T(ρ, σ) = (1/2)||ρ - σ||₁

# Details
Alternative measure of state distinguishability. Related to fidelity by T² ≤ 1 - F.
"""
function trace_distance(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    σ = (σ + σ') / 2
    diff = ρ - σ
    return 0.5 * sum(abs.(eigvals(Hermitian(diff))))
end

"""
    max_entropy(ρ::Matrix{ComplexF64})

Calculate the max-entropy (Hartley entropy) of a quantum state.

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix

# Returns
- `Float64`: Max-entropy S_max(ρ) = log₂ rank(ρ)

# Details
Depends only on the rank of the state, not the eigenvalue distribution.
"""
function max_entropy(ρ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    λs = eigvals(Hermitian(ρ))
    rank_ρ = sum(λs .> 1e-10)
    return log2(rank_ρ)
end

"""
    min_entropy(ρ::Matrix{ComplexF64})

Calculate the min-entropy of a quantum state.

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix

# Returns
- `Float64`: Min-entropy S_min(ρ) = -log₂ λ_max(ρ)

# Details
Used in quantum cryptography and randomness extraction protocols.
"""
function min_entropy(ρ::Matrix{ComplexF64})
    ρ = (ρ + ρ') / 2
    λ_max = maximum(eigvals(Hermitian(ρ)))
    return -log2(λ_max)
end

"""
    partial_trace(ρ::Matrix{ComplexF64}, dim_A::Int, dim_B::Int, subsystem::Int)

Compute the partial trace over one subsystem of a bipartite state.

# Arguments
- `ρ::Matrix{ComplexF64}`: Joint density matrix of dimension dim_A × dim_B
- `dim_A::Int`: Dimension of subsystem A
- `dim_B::Int`: Dimension of subsystem B
- `subsystem::Int`: Which subsystem to trace out (1 for A, 2 for B)

# Returns
- `Matrix{ComplexF64}`: Reduced density matrix after tracing out the specified subsystem

# Examples

# Trace out subsystem A, keep B
ρ_B = partial_trace(ρ_AB, dim_A, dim_B, 1)

# Trace out subsystem B, keep A
ρ_A = partial_trace(ρ_AB, dim_A, dim_B, 2)
"""
function partial_trace(ρ::Matrix{ComplexF64}, dim_A::Int, dim_B::Int, subsystem::Int)
    @assert size(ρ) == (dim_A * dim_B, dim_A * dim_B) "Dimension mismatch"
    ρ = (ρ + ρ') / 2
    if subsystem == 1
        ρ_B = zeros(ComplexF64, dim_B, dim_B)
        for i in 1:dim_B, j in 1:dim_B
            for k in 1:dim_A
                ρ_B[i,j] += ρ[(k-1)*dim_B+i, (k-1)*dim_B+j]
            end
        end
        return ρ_B
    else


        ρ_A = zeros(ComplexF64, dim_A, dim_A)
            for i in 1:dim_A, j in 1:dim_A
                for l in 1:dim_B
                    idx1 = (i-1)*dim_B + l
                    idx2 = (j-1)*dim_B + l
                    ρ_A[i,j] += ρ[idx1, idx2]
                end
            end
       return ρ_A
    end
end

end 
