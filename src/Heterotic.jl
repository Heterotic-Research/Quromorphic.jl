module Heterotic

# Write your package code here.


export allo

allo() = "Hello from Heterotic!"

include("QSim/QSim.jl")    # loads src/QSim/QSim.jl
include("Algorithms/grovers.jl") # loads src/Algorithms/grovers.jl
include("Algorithms/QFT.jl") # loads src/Algorithms/QFT.jl

using .QSim  
using .Grovers
using .QFT

export QSim 
export Grovers
export QFT


end
