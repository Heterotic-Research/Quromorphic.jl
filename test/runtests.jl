using Quromorphic
using Test

@testset "Quromorphic.jl" begin
    # Write your tests here.
    @testset "allo function" begin
        result = Quromorphic.allo()
        @test result == "Hello from Quromorphic!"
    end
    @testset "test_QSim.jl" begin
        include("test_QSim.jl")
    end
    @testset "test_LIF.jl" begin
        include("test_LIF.jl")
    end
end
