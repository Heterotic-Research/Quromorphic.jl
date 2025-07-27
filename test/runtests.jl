using Quromorphic
using Test

@testset "Quromorphic.jl" begin
    # Write your tests here.
    @testset "allo function" begin
        result = Quromorphic.allo()
        @test result == "Hello from Heterotic!"
    end
    @testset "test_QSim.jl" begin
        include("test_QSim.jl")
    end
end
