# Quromorphic.jl

![Julia](https://img.shields.io/badge/Julia-1.11+-9558B2?logo=julia&logoColor=white)
[![Build Status](https://github.com/Heterotic-Research/Quromorphic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Heterotic-Research/Quromorphic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Heterotic-Research/Quromorphic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Heterotic-Research/Quromorphic.jl)
[![Documentation](https://github.com/Heterotic-Research/Quromorphic.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/Heterotic-Research/Quromorphic.jl/actions/workflows/docs.yml)

**Quromorphic.jl** is a quantum neuromorphic model prototyping library written in Julia. It combines quantum computing principles with neuromorphic architectures, providing tools for designing, simulating, and evaluating hybrid quantum-neural systems. The package supports both traditional state vector simulations and tensor network-based methods (e.g., Matrix Product States), and is designed to scale efficiently for distributed computing environments.

---

##  Installation

1. Quromorphic.jl requires Julia 1.9 or later.

2. Clone this repository:
```julia
using Pkg
Pkg.add(url="https://github.com/Heterotic-Research/Quromorphic.jl")
```

3. Start Julia and instantiate the project:
   ```julia
   julia> ] activate .
   julia> instantiate
   ```
