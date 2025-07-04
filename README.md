# Heterotic

[![Build Status](https://github.com/jajapuramshivasai/Heterotic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jajapuramshivasai/Heterotic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jajapuramshivasai/Heterotic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jajapuramshivasai/Heterotic.jl)

# Heterotic.jl

**Heterotic.jl** is a high-performance, extensible Julia package for simulating quantum circuits and modeling quantum information processing tasks. It supports both traditional state vector simulations and tensor network-based methods (e.g., Matrix Product States), and is designed to scale efficiently for distributed computing environments.

<div align="center">
  <img src="https://raw.githubusercontent.com/jajapuramshivasai/Heterotic.jl/main/docs/logo.png" alt="Heterotic Logo" width="200"/>
</div>

---

## ✨ Features

- 🧠 **Two Representations**: Choose between `StateVectorRep` and `TensorNetworkRep` backends.
- ⚛️ **Custom Quantum Circuits**: Build and run quantum circuits with gates like `X`, `H`, `CNOT`, and user-defined unitaries.
- 🕸️ **Lattice & Graph-based Circuits**: Generate circuits from 1D chains and 2D lattice graphs.
- 📈 **Measurement & Probabilities**: Simulate measurements in the `X`, `Y`, `Z` bases and extract Born rule probabilities.
- 🔌 **ITensor Integration**: Efficient tensor contraction via [ITensors.jl](https://itensor.org/docs.jl/).
- 🚀 **High Performance**: Sparse matrix support, multi-threading, and distributed parallelism (WIP).

---

## 📦 Installation

Heterotic.jl requires Julia 1.9 or later.

```julia
using Pkg
Pkg.add(url="https://github.com/jajapuramshivasai/Heterotic.jl")
