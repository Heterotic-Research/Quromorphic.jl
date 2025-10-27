module Quromorphic

# Write your package code here.


export allo

allo() = "Hello from Quromorphic!"
include("Quantum/QSim.jl")    # loads src/QSim/QSim.jl
include("Neuromorphic/LSM.jl")   # loads src/Neuro/LSM.jl
include("Quantum/QInfo.jl")  # loads src/QInfo/QInfo.jl
 
using .QSim  
using .LSM
using .QInfo

export LSM
export QSim 
export QInfo


end
