module Quromorphic

# Write your package code here.


export allo

allo() = "Hello from Quromorphic!"
include("Quantum/QSim.jl")    # loads src/QSim/QSim.jl
include("Neuromorphic/LSM.jl")   # loads src/Neuro/LSM.jl
 
using .QSim  
using .LSM

export LSM
export QSim 


end
