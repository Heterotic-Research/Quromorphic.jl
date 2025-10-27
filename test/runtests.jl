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
    @testset "test_LSM.jl" begin
        include("test_LSM.jl")
    end
    @testset "test_QInfo.jl" begin
        include("test_QInfo.jl")
    end
end
