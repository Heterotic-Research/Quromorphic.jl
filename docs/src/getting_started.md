# Getting Started

This guide will help you get started with Quromorphic.jl.

## Basic Quantum Simulation

```julia
using Quromorphic.QSim

# Create a 2-qubit state in |00⟩
s = statevector(2, 0)

# Apply Hadamard to first qubit
h!(s, 1)

# Apply CNOT gate
cnot!(s, 1, 2)

# This creates a Bell state (|00⟩ + |11⟩)/√2
prstate(s)
```

## Basic Neuromorphic Computing

```julia
using Quromorphic.LSM

# Create an LSM with 3 inputs, 50 reservoir neurons, 2 outputs
lsm = LSMNet(3, 50, 2; spectral_radius=0.9, connectivity=0.2)

# Generate some data
inputs = rand(3, 100)
targets = rand(2, 100)

# Train the readout
train_readout!(lsm, inputs, targets)

# Make predictions
predictions = predict(lsm, inputs)
```

