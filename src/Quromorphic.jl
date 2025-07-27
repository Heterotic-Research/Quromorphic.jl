module Quromorphic

# Write your package code here.


export allo

allo() = "Hello from Heterotic!"
include("QSim/QSim.jl")    # loads src/QSim/QSim.jl

using .QSim  
export QSim 


end
