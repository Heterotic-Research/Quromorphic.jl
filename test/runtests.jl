using Heterotic
using Test

@testset "Heterotic.jl" begin
    # Write your tests here.
    @testset "allo function" begin
        result = Heterotic.allo()
        @test result == "Hello from Heterotic!"
    end
    @testset "test_QSim.jl" begin
        include("test_QSim.jl")
    end
    @testset "test_Algorithms.jl" begin
        include("test_Algorithms.jl")
    end
end
