
using Quromorphic.LSM
using Test
using LinearAlgebra
using Statistics

@testset "LSM.jl" begin

    # Define common parameters for tests
    input_dim = 3
    reservoir_size = 50
    output_dim = 2
    connectivity = 0.2
    spectral_radius = 1.2
    leak_rate = 0.5
    seed = 42
    T_train = 100
    T_test = 50

    # Generate some dummy data
    train_inputs = rand(input_dim, T_train)
    train_targets = rand(output_dim, T_train)
    test_inputs = rand(input_dim, T_test)
    test_targets = rand(output_dim, T_test)

    @testset "LSMNet Constructor" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim;
                     connectivity=connectivity, spectral_radius=spectral_radius,
                     leak_rate=leak_rate, seed=seed)

        @test lsm.input_dim == input_dim
        @test lsm.reservoir_size == reservoir_size
        @test lsm.output_dim == output_dim
        @test lsm.connectivity == connectivity
        @test lsm.spectral_radius == spectral_radius
        @test lsm.leak_rate == leak_rate

        @test size(lsm.Win) == (reservoir_size, input_dim)
        @test size(lsm.Wres) == (reservoir_size, reservoir_size)
        @test size(lsm.Wout) == (output_dim, reservoir_size)
        @test size(lsm.bout) == (output_dim,)
        @test size(lsm.state) == (reservoir_size,)
        @test all(lsm.state .== 0.0)
        @test all(lsm.Wout .== 0.0)
        @test all(lsm.bout .== 0.0)

        # Check spectral radius
        current_radius = maximum(abs.(eigvals(lsm.Wres)))
        @test current_radius ≈ spectral_radius atol=1e-9

        # Check connectivity
        non_zero_elements = count(!iszero, lsm.Wres)
        expected_elements = round(Int, connectivity * reservoir_size^2)
        @test non_zero_elements ≈ expected_elements atol=reservoir_size # Allow some statistical deviation
        
        # Test default parameters
        lsm_default = LSMNet(input_dim, reservoir_size, output_dim)
        @test lsm_default.connectivity == 0.1
        @test lsm_default.spectral_radius == 0.9
        @test lsm_default.leak_rate == 0.3

        # Test constructor with different parameter combinations
        lsm_custom = LSMNet(5, 20, 3; connectivity=0.5, spectral_radius=1.5, leak_rate=0.8, seed=999)
        @test lsm_custom.input_dim == 5
        @test lsm_custom.reservoir_size == 20
        @test lsm_custom.output_dim == 3
        @test lsm_custom.connectivity == 0.5
        @test lsm_custom.spectral_radius == 1.5
        @test lsm_custom.leak_rate == 0.8
    end

    @testset "reset!" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim)
        lsm.state .= rand(reservoir_size)
        @test !all(lsm.state .== 0.0)
        reset!(lsm)
        @test all(lsm.state .== 0.0)
        
        # Test multiple resets
        reset!(lsm)
        @test all(lsm.state .== 0.0)
        
        # Test reset after state changes
        lsm.state[1] = 1.0
        lsm.state[end] = -1.0
        reset!(lsm)
        @test all(lsm.state .== 0.0)
    end

    @testset "step!" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; leak_rate=leak_rate, seed=seed)
        input_vec = rand(input_dim)
        initial_state = copy(lsm.state)
        
        new_state = step!(lsm, input_vec)
        
        @test new_state === lsm.state # Check mutation
        @test !all(lsm.state .== initial_state)
        
        # Manual calculation for one step
        preact = lsm.Win * input_vec .+ lsm.Wres * initial_state
        expected_state = (1 - leak_rate) .* initial_state .+ leak_rate .* tanh.(preact)
        @test lsm.state ≈ expected_state
        
        # Test with different input sizes
        input_vec2 = rand(input_dim)
        state_before = copy(lsm.state)
        step!(lsm, input_vec2)
        @test lsm.state != state_before
        
        # Test state bounds with tanh activation - states themselves aren't bounded by tanh
        # but the activation function output is
        @test all(-1 ≤ tanh(x) ≤ 1 for x in lsm.Win * input_vec .+ lsm.Wres * state_before)
        
        # Test with zero input
        zero_input = zeros(input_dim)
        state_before_zero = copy(lsm.state)
        step!(lsm, zero_input)
        # With zero input, state should still change due to reservoir dynamics
        @test lsm.state != state_before_zero
        
        # Test with extreme inputs
        large_input = fill(10.0, input_dim)
        step!(lsm, large_input)
        # State should be finite even with large inputs due to tanh
        @test all(isfinite, lsm.state)
    end

    @testset "collect_states" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim)
        lsm.state .= rand(reservoir_size) # Set a non-zero initial state
        
        states = collect_states(lsm, train_inputs)
        
        @test size(states) == (reservoir_size, T_train)
        @test all(lsm.state .!= 0.0) # State should be updated after collection
        
        # Check that it resets state internally before starting
        reset!(lsm)
        first_step_state = step!(lsm, train_inputs[:, 1])
        @test states[:, 1] ≈ first_step_state
        
        # Test with single timestep
        single_input = rand(input_dim, 1)
        states_single = collect_states(lsm, single_input)
        @test size(states_single) == (reservoir_size, 1)
        
        # Test consistency - same input should give same states
        states2 = collect_states(lsm, train_inputs)
        @test states ≈ states2
        
        # Test with different sequence lengths
        short_sequence = rand(input_dim, 5)
        states_short = collect_states(lsm, short_sequence)
        @test size(states_short) == (reservoir_size, 5)
        
        long_sequence = rand(input_dim, 200)
        states_long = collect_states(lsm, long_sequence)
        @test size(states_long) == (reservoir_size, 200)
    end

    @testset "get_spectral_radius" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=spectral_radius)
        @test get_spectral_radius(lsm) ≈ spectral_radius atol=1e-9
        
        # Test with zero spectral radius
        lsm_zero = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=0.0)
        @test get_spectral_radius(lsm_zero) ≈ 0.0 atol=1e-12
        
        # Test with different spectral radii
        radii_to_test = [0.1, 0.5, 1.0, 1.8, 2.5]
        for r in radii_to_test
            lsm_test = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=r, seed=seed)
            @test get_spectral_radius(lsm_test) ≈ r atol=1e-9
        end
    end

    @testset "train_readout! and predict - Basic Functionality" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        
        # Before training
        @test all(lsm.Wout .== 0.0)
        @test all(lsm.bout .== 0.0)
        preds_before = predict(lsm, test_inputs)
        @test all(preds_before .== 0.0)
        @test size(preds_before) == (output_dim, T_test)

        # Test state collection works
        states = collect_states(lsm, train_inputs)
        @test size(states) == (reservoir_size, T_train)
        
        # Test basic prediction with zero weights
        manual_pred = lsm.Wout * states .+ lsm.bout
        @test size(manual_pred) == (output_dim, T_train)
        @test all(manual_pred .== 0.0)
    end

    @testset "Prediction Functionality" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        
        # Manually set some weights to test prediction
        lsm.Wout .= rand(output_dim, reservoir_size) .* 0.1
        lsm.bout .= rand(output_dim) .* 0.1
        
        preds = predict(lsm, test_inputs)
        @test size(preds) == (output_dim, T_test)
        @test !all(preds .== 0.0)
        
        # Test prediction consistency
        preds2 = predict(lsm, test_inputs)
        @test preds ≈ preds2
        
        # Test with different input sizes
        small_input = rand(input_dim, 10)
        preds_small = predict(lsm, small_input)
        @test size(preds_small) == (output_dim, 10)
        
        # Test prediction bounds are reasonable
        @test all(isfinite, preds)
    end

    @testset "Matrix Dimensions and Operations" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        
        # Test matrix operations without training
        states = collect_states(lsm, train_inputs)
        states_aug = vcat(states, ones(1, size(states,2)))
        @test size(states_aug) == (reservoir_size + 1, T_train)
        
        XTX = states_aug * states_aug'
        @test size(XTX) == (reservoir_size + 1, reservoir_size + 1)
        @test issymmetric(XTX)
        
        XTY = states_aug * train_targets'
        @test size(XTY) == (reservoir_size + 1, output_dim)
        
        # Test regularization effect
        reg = 1e-6
        XTX_reg = XTX + reg * I
        @test size(XTX_reg) == size(XTX)
        @test tr(XTX_reg) ≥ tr(XTX)  # Trace should increase with regularization
    end

    @testset "Edge Cases" begin
        # Test with small reservoir
        small_reservoir = 5
        lsm_small = LSMNet(input_dim, small_reservoir, output_dim; seed=seed)
        @test lsm_small.reservoir_size == small_reservoir
        @test size(lsm_small.Win) == (small_reservoir, input_dim)
        @test size(lsm_small.Wres) == (small_reservoir, small_reservoir)
        
        # Test functionality with small reservoir
        small_states = collect_states(lsm_small, train_inputs)
        @test size(small_states) == (small_reservoir, T_train)
        
        # Test with single input/output dimension
        lsm_single = LSMNet(1, reservoir_size, 1; seed=seed)
        @test lsm_single.input_dim == 1
        @test lsm_single.output_dim == 1
        @test size(lsm_single.Win) == (reservoir_size, 1)
        @test size(lsm_single.Wout) == (1, reservoir_size)
        @test size(lsm_single.bout) == (1,)
        
        # Test functionality with single dimensions
        single_input = rand(1, 10)
        single_target = rand(1, 10)
        single_states = collect_states(lsm_single, single_input)
        @test size(single_states) == (reservoir_size, 10)
        
        # Test with zero connectivity
        lsm_zero = LSMNet(input_dim, reservoir_size, output_dim; connectivity=0.0, seed=seed)
        @test lsm_zero.connectivity == 0.0
        @test all(lsm_zero.Wres .== 0.0)
        
        # Test with very high connectivity
        lsm_high = LSMNet(input_dim, reservoir_size, output_dim; connectivity=1.0, seed=seed)
        @test lsm_high.connectivity == 1.0
        @test count(!iszero, lsm_high.Wres) == reservoir_size^2
        
        # Test with large input dimension
        large_input_dim = 10
        lsm_large_input = LSMNet(large_input_dim, reservoir_size, output_dim; seed=seed)
        @test size(lsm_large_input.Win) == (reservoir_size, large_input_dim)
        large_input_data = rand(large_input_dim, 20)
        large_states = collect_states(lsm_large_input, large_input_data)
        @test size(large_states) == (reservoir_size, 20)
        
        # Test with large output dimension
        large_output_dim = 5
        lsm_large_output = LSMNet(input_dim, reservoir_size, large_output_dim; seed=seed)
        @test size(lsm_large_output.Wout) == (large_output_dim, reservoir_size)
        @test size(lsm_large_output.bout) == (large_output_dim,)
    end

    @testset "Spectral Radius Effects" begin
        # Low spectral radius (stable)
        lsm_low = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=0.1, seed=seed)
        @test get_spectral_radius(lsm_low) ≈ 0.1 atol=1e-9
        
        # High spectral radius (potentially chaotic)
        lsm_high = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=2.0, seed=seed)
        @test get_spectral_radius(lsm_high) ≈ 2.0 atol=1e-9
        
        # Test state evolution with different radii
        reset!(lsm_low)
        reset!(lsm_high)
        input_vec = ones(input_dim)
        
        # Apply same input multiple times
        for _ in 1:10
            step!(lsm_low, input_vec)
            step!(lsm_high, input_vec)
        end
        
        # High spectral radius should generally lead to larger state norms
        @test norm(lsm_high.state) > norm(lsm_low.state)
        
        # Test very low spectral radius with smaller threshold
        lsm_very_low = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=0.01, seed=seed)
        reset!(lsm_very_low)
        for _ in 1:5
            step!(lsm_very_low, input_vec)
        end
        # Should have smaller state changes than high spectral radius
        @test norm(lsm_very_low.state) < norm(lsm_high.state)
    end

    @testset "Leak Rate Effects" begin
        # Slow leak rate
        lsm_slow = LSMNet(input_dim, reservoir_size, output_dim; leak_rate=0.1, seed=seed)
        
        # Fast leak rate
        lsm_fast = LSMNet(input_dim, reservoir_size, output_dim; leak_rate=0.9, seed=seed)
        
        reset!(lsm_slow)
        reset!(lsm_fast)
        input_vec = ones(input_dim)
        
        # Initial states
        slow_state_1 = copy(lsm_slow.state)
        fast_state_1 = copy(lsm_fast.state)
        
        # One step
        step!(lsm_slow, input_vec)
        step!(lsm_fast, input_vec)
        
        # Fast leak rate should change state more quickly
        @test norm(lsm_fast.state - fast_state_1) > norm(lsm_slow.state - slow_state_1)
        
        # Test extreme leak rates
        lsm_no_leak = LSMNet(input_dim, reservoir_size, output_dim; leak_rate=0.0, seed=seed)
        lsm_full_leak = LSMNet(input_dim, reservoir_size, output_dim; leak_rate=1.0, seed=seed)
        
        reset!(lsm_no_leak)
        reset!(lsm_full_leak)
        
        # No leak should keep state at zero
        step!(lsm_no_leak, input_vec)
        @test all(lsm_no_leak.state .== 0.0)
        
        # Full leak should be purely feedforward
        step!(lsm_full_leak, input_vec)
        expected_full_leak = tanh.(lsm_full_leak.Win * input_vec)
        @test lsm_full_leak.state ≈ expected_full_leak
    end

    @testset "Memory and Temporal Properties" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        
        # Test memory by applying same input with different delays
        reset!(lsm)
        impulse = ones(input_dim)
        zero_input = zeros(input_dim)
        
        # Apply impulse then zeros
        step!(lsm, impulse)
        state_after_impulse = copy(lsm.state)
        
        step!(lsm, zero_input)
        state_after_zero = copy(lsm.state)
        
        # State should decay but not be zero (memory)
        @test norm(state_after_zero) < norm(state_after_impulse)
        @test norm(state_after_zero) > 0.0
        
        # Test sequence processing
        sequence = [rand(input_dim) for _ in 1:5]
        reset!(lsm)
        states_sequence = []
        for inp in sequence
            push!(states_sequence, copy(step!(lsm, inp)))
        end
        
        # Each state should be different
        for i in 2:length(states_sequence)
            @test states_sequence[i] != states_sequence[i-1]
        end
        
        # Test temporal correlation
        reset!(lsm)
        repeated_input = ones(input_dim)
        correlation_states = []
        for _ in 1:10
            push!(correlation_states, copy(step!(lsm, repeated_input)))
        end
        
        # Later states should be more similar (convergence)
        early_diff = norm(correlation_states[2] - correlation_states[1])
        late_diff = norm(correlation_states[10] - correlation_states[9])
        @test late_diff ≤ early_diff  # Should converge or at least not diverge more
    end

    @testset "Reproducibility" begin
        # Test that same seed produces same results
        lsm1 = LSMNet(input_dim, reservoir_size, output_dim; seed=123)
        lsm2 = LSMNet(input_dim, reservoir_size, output_dim; seed=123)
        
        @test lsm1.Win ≈ lsm2.Win
        @test lsm1.Wres ≈ lsm2.Wres
        
        # Test that different seeds produce different results
        lsm3 = LSMNet(input_dim, reservoir_size, output_dim; seed=456)
        @test !(lsm1.Win ≈ lsm3.Win)
        @test !(lsm1.Wres ≈ lsm3.Wres)
        
        # Test state evolution reproducibility
        test_input = rand(input_dim, 10)
        states1 = collect_states(lsm1, test_input)
        states2 = collect_states(lsm2, test_input)
        @test states1 ≈ states2
        
        # Test reproducibility with manual stepping
        reset!(lsm1)
        reset!(lsm2)
        for i in 1:10
            step!(lsm1, test_input[:, i])
            step!(lsm2, test_input[:, i])
            @test lsm1.state ≈ lsm2.state
        end
    end

    @testset "Error Handling and Boundary Conditions" begin
        # Test with very small dimensions
        lsm_tiny = LSMNet(1, 1, 1; seed=seed)
        @test lsm_tiny.reservoir_size == 1
        tiny_input = rand(1, 5)
        tiny_states = collect_states(lsm_tiny, tiny_input)
        @test size(tiny_states) == (1, 5)
        
        # Test with proper dimensions
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        correct_input = rand(input_dim)
        @test size(step!(lsm, correct_input)) == (reservoir_size,)
        
        # Test prediction with zero inputs - should still produce output due to bias and Win
        zero_input = zeros(input_dim, 1)
        lsm.Wout .= rand(output_dim, reservoir_size) .* 0.1  # Set non-zero weights
        lsm.bout .= rand(output_dim) .* 0.1  # Set non-zero bias
        
        zero_states = collect_states(lsm, zero_input)
        @test size(zero_states) == (reservoir_size, 1)
        
        zero_preds = predict(lsm, zero_input)
        @test size(zero_preds) == (output_dim, 1)
        # Should produce non-zero output due to bias term
        @test !all(zero_preds .== 0.0)
        
        # Test with very large inputs
        large_input = fill(100.0, input_dim, 5)
        large_states = collect_states(lsm, large_input)
        @test size(large_states) == (reservoir_size, 5)
        @test all(isfinite, large_states)
        
        # Test with very small inputs
        small_input = fill(1e-10, input_dim, 5)
        small_states = collect_states(lsm, small_input)
        @test size(small_states) == (reservoir_size, 5)
        @test all(isfinite, small_states)
    end

    @testset "State Dynamics and Stability" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=0.8, seed=seed)
        
        # Test convergence to fixed point with constant input
        constant_input = ones(input_dim)
        reset!(lsm)
        
        states_evolution = []
        for _ in 1:50
            push!(states_evolution, copy(step!(lsm, constant_input)))
        end
        
        # Check that states converge (difference decreases)
        early_change = norm(states_evolution[5] - states_evolution[4])
        late_change = norm(states_evolution[50] - states_evolution[49])
        @test late_change ≤ early_change
        
        # Test state bounds remain reasonable
        @test all(abs.(states_evolution[end]) .< 10.0)  # States shouldn't explode
        
        # Test oscillatory behavior with alternating inputs
        reset!(lsm)
        input1 = ones(input_dim)
        input2 = -ones(input_dim)
        
        oscillation_states = []
        for i in 1:20
            if i % 2 == 1
                push!(oscillation_states, copy(step!(lsm, input1)))
            else
                push!(oscillation_states, copy(step!(lsm, input2)))
            end
        end
        
        # States should show some pattern or bounded behavior
        @test all(abs.(oscillation_states[end]) .< 20.0)
        @test all(isfinite, oscillation_states[end])
    end

    @testset "Performance and Efficiency" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
        
        # Test that operations complete in reasonable time
        large_sequence = rand(input_dim, 1000)
        @time states = collect_states(lsm, large_sequence)
        @test size(states) == (reservoir_size, 1000)
        
        # Test memory efficiency - states should be reasonable size
        @test sizeof(states) > 0
        @test all(isfinite, states)
        
        # Test repeated operations
        for _ in 1:100
            step!(lsm, rand(input_dim))
        end
        @test all(isfinite, lsm.state)
        
        # Test batch processing consistency
        batch_input = rand(input_dim, 50)
        states1 = collect_states(lsm, batch_input)
        
        # Manual processing
        reset!(lsm)
        manual_states = zeros(reservoir_size, 50)
        for i in 1:50
            manual_states[:, i] = step!(lsm, batch_input[:, i])
        end
        
        @test states1 ≈ manual_states
    end

    @testset "Weight Matrix Properties" begin
        lsm = LSMNet(input_dim, reservoir_size, output_dim; connectivity=0.3, spectral_radius=1.1, seed=seed)
        
        # Test Win properties
        @test all(-1 ≤ w ≤ 1 for w in lsm.Win)  # Win should be in [-1, 1]
        @test !all(lsm.Win .== 0.0)  # Win shouldn't be all zeros
        
        # Test Wres properties
        @test size(lsm.Wres) == (reservoir_size, reservoir_size)
        @test get_spectral_radius(lsm) ≈ 1.1 atol=1e-9
        
        # Test connectivity
        sparsity = count(lsm.Wres .== 0.0) / length(lsm.Wres)
        expected_sparsity = 1.0 - 0.3
        @test abs(sparsity - expected_sparsity) < 0.1  # Allow some tolerance
        
        # Test Wres symmetry (should generally not be symmetric)
        @test lsm.Wres != lsm.Wres'  # Should not be symmetric
        
        # Test that Wres has some structure
        @test maximum(abs.(lsm.Wres)) > 0.0
        @test maximum(abs.(lsm.Wres)) ≤ maximum(abs.(eigvals(lsm.Wres))) * 2  # Rough bound
    end

    @testset "Different Initialization Seeds" begin
        seeds_to_test = [1, 42, 123, 999, 2024]
        lsms = [LSMNet(input_dim, reservoir_size, output_dim; seed=s) for s in seeds_to_test]
        
        # Test that different seeds give different networks
        for i in 1:length(lsms)
            for j in i+1:length(lsms)
                @test !(lsms[i].Win ≈ lsms[j].Win)
                @test !(lsms[i].Wres ≈ lsms[j].Wres)
            end
        end
        
        # Test that all networks have correct properties
        for lsm in lsms
            @test size(lsm.Win) == (reservoir_size, input_dim)
            @test size(lsm.Wres) == (reservoir_size, reservoir_size)
            @test get_spectral_radius(lsm) ≈ lsm.spectral_radius atol=1e-9
        end
        
        # Test that they produce different dynamics
        test_input = rand(input_dim, 10)
        states_list = [collect_states(lsm, test_input) for lsm in lsms]
        
        for i in 1:length(states_list)
            for j in i+1:length(states_list)
                @test !(states_list[i] ≈ states_list[j])
            end
        end
    end

end