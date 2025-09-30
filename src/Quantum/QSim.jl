module QSim

using LinearAlgebra

export statevector,cnot!,measure_z,swap!, u2!, h!, x!, y!, z!, rx!, ry!, rz!, crx!, cry!, crz!, mp, prstate, measure_x,measure_y, density_matrix,nb,id,statevector_to_density_matrix ,X,Y,Z,I2

# Core Constants 
const c1 = ComplexF64(1)
const im_F64 = ComplexF64(0, 1)
const I2 = Matrix{ComplexF64}([1 0; 0 1])

const H = ComplexF64(1/√2) * Matrix{ComplexF64}([1 1; 1 -1])
const X = Matrix{ComplexF64}([0 1; 1 0])
const Y = Matrix{ComplexF64}([0 -im_F64; im_F64 0])
const Z = Matrix{ComplexF64}([1 0; 0 -1])
const P0 = Matrix{ComplexF64}([1 0; 0 0])
const P1 = Matrix{ComplexF64}([0 0; 0 1])

# Precomputed rotation gates for common angles
const RX_PI_2 = ComplexF64(cos(π/4))*I2 - im_F64*ComplexF64(sin(π/4))*X
const RY_PI_2 = ComplexF64(cos(π/4))*I2 - im_F64*ComplexF64(sin(π/4))*Y
const RZ_PI_2 = ComplexF64(cos(π/4))*I2 - im_F64*ComplexF64(sin(π/4))*Z

# Rotation gate caches for performance
const RX_CACHE = Dict{Float64, Matrix{ComplexF64}}()
const RY_CACHE = Dict{Float64, Matrix{ComplexF64}}()
const RZ_CACHE = Dict{Float64, Matrix{ComplexF64}}()

function id(n::Int)
    n == 0 ? Matrix{ComplexF64}([c1;;]) : Diagonal(ones(ComplexF64, 1 << n))
end

function statevector(n::Int, m::Int)
    s = zeros(ComplexF64, 1 << n)
    s[m+1] = c1 
    s
end

function statevector_to_density_matrix(s::Vector{ComplexF64})
    return s * s'
end

function density_matrix(n::Int, m::Int)
    s = statevector(n, m)
    return statevector_to_density_matrix(s)
end

nb(s::Vector{ComplexF64}) = round(Int, log2(length(s)))
nb(rho::Matrix{ComplexF64}) = round(Int, log2(size(rho, 1)))


function u2!(s::Vector{ComplexF64}, t::Int, U::Matrix{ComplexF64})
    q = nb(s)
    t <= q || error("Qubit index out of range")
    
    l = t - 1
    r = q - t
    op = kron(id(r), kron(U, id(l)))
    temp = similar(s)
    mul!(temp, op, s)
    copyto!(s, temp)
    s
end



function u2!(rho::Matrix{ComplexF64}, t::Int, U::Matrix{ComplexF64})
    q = round(Int, log2(size(rho,1)))
    t <= q || error("Qubit index out of range")
    
    l = t - 1
    r = q - t
    op = kron(id(r), kron(U, id(l)))
    
    temp = similar(rho)
    mul!(temp, op, rho)
    mul!(rho, temp, op')
    rho
end

function h!(s::Vector{ComplexF64}, t::Int)
    u2!(s, t, H)
end

function h!(rho::Matrix{ComplexF64}, t::Int)
    u2!(rho, t, H)
end

function x!(s::Vector{ComplexF64}, t::Int)
    u2!(s, t, X)
end

function x!(rho::Matrix{ComplexF64}, t::Int)
    u2!(rho, t, X)
end

function y!(s::Vector{ComplexF64}, t::Int)
    u2!(s, t, Y)
end

function y!(rho::Matrix{ComplexF64}, t::Int)
    u2!(rho, t, Y)
end

function z!(s::Vector{ComplexF64}, t::Int)
    u2!(s, t, Z)
end

function z!(rho::Matrix{ComplexF64}, t::Int)
    u2!(rho, t, Z)
end

# Rotation gates with caching
function rx_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RX_CACHE, theta_f64)
        return RX_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*X
    RX_CACHE[theta_f64] = gate
    return gate
end

function ry_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RY_CACHE, theta_f64)
        return RY_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*Y
    RY_CACHE[theta_f64] = gate
    return gate
end

function rz_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RZ_CACHE, theta_f64)
        return RZ_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*Z
    RZ_CACHE[theta_f64] = gate
    return gate
end

function rx!(s::Vector{ComplexF64}, t::Int, theta::Real)
    u2!(s, t, rx_gate(theta))
end

function rx!(rho::Matrix{ComplexF64}, t::Int, theta::Real)
    u2!(rho, t, rx_gate(theta))
end

function ry!(s::Vector{ComplexF64}, t::Int, theta::Real)
    u2!(s, t, ry_gate(theta))
end

function ry!(rho::Matrix{ComplexF64}, t::Int, theta::Real)
    u2!(rho, t, ry_gate(theta))
end

function rz!(s::Vector{ComplexF64}, t::Int, theta::Real)
    u2!(s, t, rz_gate(theta))
end

function rz!(rho::Matrix{ComplexF64}, t::Int, theta::Real)
    u2!(rho, t, rz_gate(theta))
end

# Controlled gates
function controlled!(s::Vector{ComplexF64}, c::Int, t::Int, V::Matrix{ComplexF64})
    q = nb(s)
    c <= q && t <= q || error("Qubit index out of range")
    c == t && error("Control and target cannot be the same qubit")
    
    # Identify lower and higher indices
    a = min(c, t)
    b = max(c, t)

    # Build identity operators for regions outside control/target pair
    left = id(q - b)             
    right = id(a - 1)            
    mid = (b - a - 1) > 0 ? id(b - a - 1) : Matrix{ComplexF64}([c1;;]) 

    # Construct two-qubit controlled operator
    U2 = if c < t
        kron(id(1), P0) + kron(V, P1)
    else
        kron(P0, id(1)) + kron(P1, V)
    end

    # Apply full operator efficiently
    op = kron(left, kron(U2, kron(mid, right)))
    temp = similar(s)
    mul!(temp, op, s)
    copyto!(s, temp)
    
    return s
end

function cnot!(s::Vector{ComplexF64}, c::Int, t::Int)
    controlled!(s, c, t, X)
end

function controlled!(rho::Matrix{ComplexF64}, c::Int, t::Int, V::Matrix{ComplexF64})
    q = round(Int, log2(size(rho, 1)))
    c <= q && t <= q || error("Qubit index out of range")
    c == t && error("Control and target cannot be the same qubit")
    
    a = min(c, t)
    b = max(c, t)
    left = id(q - b)
    right = id(a - 1)
    mid = (b - a - 1) > 0 ? id(b - a - 1) : reshape([c1], 1, 1)
    
    U2 = if c < t
        kron(id(1), P0) + kron(V, P1)
    else
        kron(P0, id(1)) + kron(P1, V)
    end
    
    op = kron(left, kron(U2, kron(mid, right)))
    temp = similar(rho)
    mul!(temp, op, rho)
    mul!(rho, temp, op')
    
    rho
end

function cnot!(rho::Matrix{ComplexF64}, c::Int, t::Int)
    controlled!(rho, c, t, X)
end

function crx!(s::Vector{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rx_gate(theta))
end

function crx!(rho::Matrix{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, rx_gate(theta))
end

function cry!(s::Vector{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, ry_gate(theta))
end

function cry!(rho::Matrix{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, ry_gate(theta))
end

function crz!(s::Vector{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rz_gate(theta))
end

function crz!(rho::Matrix{ComplexF64}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, rz_gate(theta))
end

function swap!(s::Vector{ComplexF64}, q1::Int, q2::Int)
    q = nb(s)
    q1 <= q && q2 <= q || error("Qubit index out of range")
    
    if q1 == q2
        return s
    end
    
    cnot!(s, q1, q2)
    cnot!(s, q2, q1)
    cnot!(s, q1, q2)
    
    s
end

function swap!(rho::Matrix{ComplexF64}, q1::Int, q2::Int)
    q = round(Int, log2(size(rho, 1)))
    q1 <= q && q2 <= q || error("Qubit index out of range")
    
    if q1 == q2
        return rho
    end
    
    cnot!(rho, q1, q2)
    cnot!(rho, q2, q1)
    cnot!(rho, q1, q2)
    
    rho
end

# Measurement probability functions
mp(s::Vector{ComplexF64}) = abs2.(s)

mp(rho::Matrix{ComplexF64}) = real.(diag(rho))

function measure_z(s::Vector{ComplexF64}, t::Int)
    n = nb(s)
    t <= n || error("Qubit index out of range")
    
    probs = mp(s)
    p0 = 0.0
    p1 = 0.0
    t_mask = 1 << (t-1)  # Precomputed bit mask
    
    for i in 0:(length(probs)-1)
        prob = probs[i+1]
        if prob > 1e-10
            if (i & t_mask) == 0
                p0 += prob
            else
                p1 += prob
            end
        end
    end
    
    return (p0, p1)
end

function measure_x(s::Vector{ComplexF64}, t::Int)
    s_copy = copy(s)
    h!(s_copy, t)
    return measure_z(s_copy, t)
end

function measure_y(s::Vector{ComplexF64}, t::Int)
    s_copy = copy(s)
    rx!(s_copy, t, π/2)
    return measure_z(s_copy, t)
end

function measure_z(rho::Matrix{ComplexF64}, t::Int)
    q = round(Int, log2(size(rho, 1)))
    t <= q || error("Qubit index out of range")
    
    probs = real.(diag(rho))
    p0 = 0.0
    p1 = 0.0
    t_mask = 1 << (t - 1)
    
    for i in 0:(length(probs) - 1)
        if probs[i + 1] > 1e-10
            if (i & t_mask) == 0
                p0 += probs[i + 1]
            else
                p1 += probs[i + 1]
            end
        end
    end
    
    return (p0, p1)
end

function measure_x(rho::Matrix{ComplexF64}, t::Int)
    rho_copy = copy(rho)
    h!(rho_copy, t)
    return measure_z(rho_copy, t)
end

function measure_y(rho::Matrix{ComplexF64}, t::Int)
    rho_copy = copy(rho)
    rx!(rho_copy, t, π / 2)
    return measure_z(rho_copy, t)
end


function prstate(s::Vector{ComplexF64})
    n = nb(s)
    for i in 0:(length(s)-1)
        a = s[i+1]
        abs(a) > 1e-10 && println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
    end
end

function prstate(x::Matrix{ComplexF64})
    q = round(Int, log2(size(x, 1)))
    probs = mp(x)
    for i in 0:(size(x, 1) - 1)
        p = probs[i + 1]
        abs(p) > 1e-10 && println("|", lpad(string(i, base=2), q, '0'), "⟩ : ", p)
    end
    nothing
end

end 