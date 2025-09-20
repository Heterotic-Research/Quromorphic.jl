module Quromorphic

# Write your package code here.


export allo

allo() = "Hello from Quromorphic!"
include("QSim/QSim.jl")    # loads src/QSim/QSim.jl
include("Neuro/LSM.jl")   # loads src/Neuro/LSM.jl
 
using .QSim  
using .LSM

export LSM
export QSim 


end
