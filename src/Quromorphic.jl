module Quromorphic

# Write your package code here.


export allo

allo() = "Hello from Quromorphic!"
include("Quantum/QSim.jl")    # loads src/QSim/QSim.jl
include("Neuromorphic/LIF.jl")   # loads src/Neuro/LSM.jl
 
using .QSim  
using .LIF

export LIF
export QSim 


end
