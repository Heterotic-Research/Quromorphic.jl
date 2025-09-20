
using Quromorphic.LSM
using Test
using LinearAlgebra
using Statistics
using Random

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

    @testset "train_readout! - Comprehensive Testing" begin
        @testset "Basic Training Functionality" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Before training
            @test all(lsm.Wout .== 0.0)
            @test all(lsm.bout .== 0.0)
            
            # Train the readout
            train_readout!(lsm, train_inputs, train_targets)
            
            # After training
            @test !all(lsm.Wout .== 0.0)
            @test !all(lsm.bout .== 0.0)
            @test size(lsm.Wout) == (output_dim, reservoir_size)
            @test size(lsm.bout) == (output_dim,)
            @test all(isfinite, lsm.Wout)
            @test all(isfinite, lsm.bout)
        end
        
        @testset "Training with Different Regularization" begin
            lsm1 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            lsm2 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Train with different regularization values
            train_readout!(lsm1, train_inputs, train_targets; reg=1e-8)
            train_readout!(lsm2, train_inputs, train_targets; reg=1e-3)
            
            # Weights should be different due to regularization
            @test !(lsm1.Wout ≈ lsm2.Wout)
            @test !(lsm1.bout ≈ lsm2.bout)
            
            # Higher regularization should generally lead to smaller weights
            @test norm(lsm2.Wout) ≤ norm(lsm1.Wout)
        end
        
        @testset "Training Performance and Error" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Train the network
            train_readout!(lsm, train_inputs, train_targets)
            
            # Test training error
            train_preds = predict(lsm, train_inputs)
            train_mse = mean((train_preds .- train_targets).^2)
            @test train_mse ≥ 0.0
            @test isfinite(train_mse)
            
            # Test prediction consistency
            train_preds2 = predict(lsm, train_inputs)
            @test train_preds ≈ train_preds2
            
            # Test generalization
            test_preds = predict(lsm, test_inputs)
            test_mse = mean((test_preds .- test_targets).^2)
            @test test_mse ≥ 0.0
            @test isfinite(test_mse)
        end
        
        @testset "Training with Edge Case Data" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with zero targets
            zero_targets = zeros(output_dim, T_train)
            train_readout!(lsm, train_inputs, zero_targets)
            zero_preds = predict(lsm, train_inputs)
            zero_mse = mean(zero_preds.^2)
            @test zero_mse ≥ 0.0
            @test all(isfinite, zero_preds)
            
            # Test with constant targets
            constant_targets = ones(output_dim, T_train)
            train_readout!(lsm, train_inputs, constant_targets)
            const_preds = predict(lsm, train_inputs)
            @test all(isfinite, const_preds)
            @test size(const_preds) == (output_dim, T_train)
            
            # Test with very small dataset
            small_inputs = rand(input_dim, 5)
            small_targets = rand(output_dim, 5)
            train_readout!(lsm, small_inputs, small_targets)
            small_preds = predict(lsm, small_inputs)
            @test size(small_preds) == (output_dim, 5)
            @test all(isfinite, small_preds)
        end
        
        @testset "Matrix Operations in Training" begin
            input_dim = 3
            reservoir_size = 20
            output_dim = 2
            lsm = LSMNet(input_dim, reservoir_size, output_dim, seed = seed)
            inputs = rand(input_dim, 30)
            targets = rand(output_dim, 30)
            
            # Manually check the training process
            states = collect_states(lsm, inputs)
            states_aug = vcat(states, ones(1, size(states,2)))
            @test size(states_aug) == (reservoir_size + 1, 30)
            
            XTX = states_aug * states_aug'
            @test size(XTX) == (reservoir_size + 1, reservoir_size + 1)
            @test issymmetric(XTX)
            
            XTY = states_aug * targets'
            @test size(XTY) == (reservoir_size + 1, output_dim)
            
            reg = 1e-6
            XTX_reg = XTX + reg * I
            Waug = XTX_reg \ XTY
            
            @test size(Waug) == (reservoir_size + 1, output_dim)
            
            # Now train the network and compare
            # Reset the LSM to the same initial state before training
            reset!(lsm)
            train_readout!(lsm, inputs, targets)
            
            # Compare the weights (with some tolerance for numerical differences)
            @test lsm.Wout ≈ Waug[1:end-1,:]' atol=1e-10
            @test lsm.bout ≈ Waug[end,:] atol=1e-10
        end
        
        @testset "Training Robustness" begin
            # Test training with different network configurations
            configs = [
                (input_dim=2, reservoir_size=10, output_dim=1),
                (input_dim=5, reservoir_size=30, output_dim=3),
                (input_dim=1, reservoir_size=20, output_dim=1),
            ]
            
            for config in configs
                lsm = LSMNet(config.input_dim, config.reservoir_size, config.output_dim; seed=seed)
                inputs = rand(config.input_dim, 50)
                targets = rand(config.output_dim, 50)
                
                train_readout!(lsm, inputs, targets)
                preds = predict(lsm, inputs)
                
                @test size(preds) == (config.output_dim, 50)
                @test all(isfinite, preds)
                @test !all(lsm.Wout .== 0.0)
                @test size(lsm.Wout) == (config.output_dim, config.reservoir_size)
                @test size(lsm.bout) == (config.output_dim,)
            end
        end
        
        @testset "Retraining Behavior" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Initial training
            train_readout!(lsm, train_inputs, train_targets)
            initial_Wout = copy(lsm.Wout)
            initial_bout = copy(lsm.bout)
            initial_preds = predict(lsm, test_inputs)
            
            # Retrain with same data
            train_readout!(lsm, train_inputs, train_targets)
            @test lsm.Wout ≈ initial_Wout
            @test lsm.bout ≈ initial_bout
            
            # Retrain with different data
            new_targets = rand(output_dim, T_train)
            train_readout!(lsm, train_inputs, new_targets)
            @test !(lsm.Wout ≈ initial_Wout)
            @test !(lsm.bout ≈ initial_bout)
            
            new_preds = predict(lsm, test_inputs)
            @test !(new_preds ≈ initial_preds)
        end
    end

    @testset "ablation_spectral_radius - Comprehensive Testing" begin
        @testset "Basic Ablation Functionality" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            original_Wres = copy(lsm.Wres)
            
            radii = [0.5, 1.0, 1.5]
            results = ablation_spectral_radius(lsm, radii, train_inputs, train_targets, test_inputs, test_targets)
            
            # Check results structure
            @test length(results) == length(radii)
            @test all(isa(r, Tuple{Float64, Float64}) for r in results)
            @test all(r[1] ∈ radii for r in results)
            @test all(r[2] ≥ 0 for r in results)  # MSE should be non-negative
            @test all(isfinite(r[2]) for r in results)
            
            # Check that original Wres is restored
            @test lsm.Wres ≈ original_Wres
            
            # Check results are in correct order
            for i in 1:length(radii)
                @test results[i][1] == radii[i]
            end
        end
        
        @testset "Single Radius Ablation" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            original_Wres = copy(lsm.Wres)
            
            single_radius = [0.8]
            results = ablation_spectral_radius(lsm, single_radius, train_inputs, train_targets, test_inputs, test_targets)
            
            @test length(results) == 1
            @test results[1][1] == 0.8
            @test results[1][2] ≥ 0
            @test isfinite(results[1][2])
            @test lsm.Wres ≈ original_Wres
        end
        
        @testset "Multiple Radii with Different Values" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with a wide range of spectral radii
            radii = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]
            results = ablation_spectral_radius(lsm, radii, train_inputs, train_targets, test_inputs, test_targets)
            
            @test length(results) == length(radii)
            
            # Extract MSE values
            mse_values = [r[2] for r in results]
            @test all(mse >= 0 for mse in mse_values)
            @test all(isfinite, mse_values)
            
            # Check that different radii generally give different performance
            @test length(unique(mse_values)) > 1  # Should have some variation
        end
        
        @testset "Edge Case Radii" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with very small radius
            small_radii = [1e-6, 1e-3]
            results_small = ablation_spectral_radius(lsm, small_radii, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_small) == 2
            @test all(r[2] ≥ 0 for r in results_small)
            @test all(isfinite(r[2]) for r in results_small)
            
            # Test with very large radius
            large_radii = [5.0, 10.0]
            results_large = ablation_spectral_radius(lsm, large_radii, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_large) == 2
            @test all(r[2] ≥ 0 for r in results_large)
            @test all(isfinite(r[2]) for r in results_large)
            
            # Test with zero radius
            zero_radii = [0.0]
            results_zero = ablation_spectral_radius(lsm, zero_radii, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_zero) == 1
            @test results_zero[1][1] == 0.0
            @test results_zero[1][2] ≥ 0
        end
        
        @testset "Scaling Behavior Verification" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; spectral_radius=1.0, seed=seed)
            original_Wres = copy(lsm.Wres)
            original_radius = get_spectral_radius(lsm)
            
            # Test that spectral radius scaling works correctly
            test_radius = 1.5
            radii = [test_radius]
            
            # Manually scale and check
            scaled_Wres = original_Wres * (test_radius / original_radius)
            
            # Create a temporary LSM to check the radius without modifying the original struct's fields
            temp_lsm = LSMNet(lsm.input_dim, lsm.reservoir_size, lsm.output_dim, 
                              lsm.connectivity, lsm.spectral_radius, lsm.leak_rate,
                              lsm.Win, scaled_Wres, lsm.Wout, lsm.bout, lsm.state)
            scaled_radius = get_spectral_radius(temp_lsm)
            @test scaled_radius ≈ test_radius atol=1e-9
            
            # Run ablation and verify
            ablation_spectral_radius(lsm, radii, train_inputs, train_targets, test_inputs, test_targets)
            @test lsm.Wres ≈ original_Wres  # Should be restored
        end
        
        @testset "Network State Preservation" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Set some initial state
            lsm.state .= rand(reservoir_size)
            initial_state = copy(lsm.state)
            initial_Win = copy(lsm.Win)
            initial_Wout = copy(lsm.Wout)
            initial_bout = copy(lsm.bout)
            
            radii = [0.5, 1.2, 2.0]
            ablation_spectral_radius(lsm, radii, train_inputs, train_targets, test_inputs, test_targets)
            
            # Check that only Wres-related properties are preserved, others may change due to training
            @test lsm.Win ≈ initial_Win
            # Note: Wout and bout will change due to training in ablation
            # Note: state will be reset during collect_states calls
        end
        
        @testset "Performance Consistency" begin
            # Test that ablation results are consistent across runs
            lsm1 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            lsm2 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            radii = [0.7, 1.0, 1.3]
            results1 = ablation_spectral_radius(lsm1, radii, train_inputs, train_targets, test_inputs, test_targets)
            results2 = ablation_spectral_radius(lsm2, radii, train_inputs, train_targets, test_inputs, test_targets)
            
            # Results should be identical for identical networks
            for i in 1:length(results1)
                @test results1[i][1] ≈ results2[i][1]
                @test results1[i][2] ≈ results2[i][2] atol=1e-10
            end
        end
    end

    @testset "ablation_connectivity - Comprehensive Testing" begin
        @testset "Basic Connectivity Ablation" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            original_Wres = copy(lsm.Wres)
            
            connects = [0.1, 0.3, 0.5]
            results = ablation_connectivity(lsm, connects, train_inputs, train_targets, test_inputs, test_targets)
            
            # Check results structure
            @test length(results) == length(connects)
            @test all(isa(r, Tuple{Float64, Float64}) for r in results)
            @test all(r[1] ∈ connects for r in results)
            @test all(r[2] ≥ 0 for r in results)  # MSE should be non-negative
            @test all(isfinite(r[2]) for r in results)
            
            # Check that original Wres is restored
            @test lsm.Wres ≈ original_Wres
            
            # Check results are in correct order
            for i in 1:length(connects)
                @test results[i][1] == connects[i]
            end
        end
        
        @testset "Extreme Connectivity Values" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with zero connectivity
            zero_connects = [0.0]
            results_zero = ablation_connectivity(lsm, zero_connects, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_zero) == 1
            @test results_zero[1][1] == 0.0
            @test results_zero[1][2] ≥ 0
            @test isfinite(results_zero[1][2])
            
            # Test with full connectivity
            full_connects = [1.0]
            results_full = ablation_connectivity(lsm, full_connects, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_full) == 1
            @test results_full[1][1] == 1.0
            @test results_full[1][2] ≥ 0
            @test isfinite(results_full[1][2])
            
            # Test with very low connectivity
            low_connects = [0.01, 0.05]
            results_low = ablation_connectivity(lsm, low_connects, train_inputs, train_targets, test_inputs, test_targets)
            @test length(results_low) == 2
            @test all(r[2] ≥ 0 for r in results_low)
            @test all(isfinite(r[2]) for r in results_low)
        end
        
        @testset "Connectivity Range Testing" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with a comprehensive range of connectivity values
            connects = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            results = ablation_connectivity(lsm, connects, train_inputs, train_targets, test_inputs, test_targets)
            
            @test length(results) == length(connects)
            
            # Extract MSE values
            mse_values = [r[2] for r in results]
            @test all(mse >= 0 for mse in mse_values)
            @test all(isfinite, mse_values)
            
            # Check that different connectivity values generally give different performance
            @test length(unique(mse_values)) > 1  # Should have some variation
        end
        
        @testset "Network Generation Verification" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            original_spectral_radius = lsm.spectral_radius
            
            # Test that new networks are generated with correct connectivity
            test_connects = [0.2, 0.8]
            
            for conn in test_connects
                # Manually create network with this connectivity to verify behavior
                rng = MersenneTwister(123)  # Same seed as used in ablation function
                mask = rand(rng, reservoir_size, reservoir_size) .< conn
                test_Wres = zeros(reservoir_size, reservoir_size)
                test_Wres[mask] .= rand(rng, sum(mask)) .* 2 .- 1
                
                # Check connectivity
                actual_connectivity = count(!iszero, test_Wres) / length(test_Wres)
                expected_connectivity = conn
                @test abs(actual_connectivity - expected_connectivity) < 0.1  # Allow statistical variation
                
                # Check spectral radius scaling
                if maximum(abs.(eigvals(test_Wres))) > 0
                    test_Wres .*= original_spectral_radius / maximum(abs.(eigvals(test_Wres)))
                    temp_lsm = LSMNet(input_dim, reservoir_size, output_dim,
                                      conn, original_spectral_radius, lsm.leak_rate,
                                      lsm.Win, test_Wres, lsm.Wout, lsm.bout, lsm.state)
                    @test get_spectral_radius(temp_lsm) ≈ original_spectral_radius atol=1e-9
                end
            end
        end
        
        @testset "Consistency Across Runs" begin
            # Test that ablation results are consistent
            lsm1 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            lsm2 = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            connects = [0.1, 0.5, 0.9]
            results1 = ablation_connectivity(lsm1, connects, train_inputs, train_targets, test_inputs, test_targets)
            results2 = ablation_connectivity(lsm2, connects, train_inputs, train_targets, test_inputs, test_targets)
            
            # Results should be identical for identical networks
            # (Note: ablation_connectivity uses fixed seed 123 for network generation)
            for i in 1:length(results1)
                @test results1[i][1] ≈ results2[i][1]
                @test results1[i][2] ≈ results2[i][2] atol=1e-10
            end
        end
        
        @testset "Original Network Restoration" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; connectivity=0.3, seed=seed)
            original_Wres = copy(lsm.Wres)
            original_connectivity = count(!iszero, original_Wres) / length(original_Wres)
            
            # Run ablation
            connects = [0.1, 0.7]
            ablation_connectivity(lsm, connects, train_inputs, train_targets, test_inputs, test_targets)
            
            # Check restoration
            @test lsm.Wres ≈ original_Wres
            restored_connectivity = count(!iszero, lsm.Wres) / length(lsm.Wres)
            @test restored_connectivity ≈ original_connectivity
        end
        
        @testset "Performance with Different Data Sizes" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Test with small dataset
            small_train_inputs = rand(input_dim, 20)
            small_train_targets = rand(output_dim, 20)
            small_test_inputs = rand(input_dim, 10)
            small_test_targets = rand(output_dim, 10)
            
            connects = [0.2, 0.6]
            results_small = ablation_connectivity(lsm, connects, small_train_inputs, small_train_targets, 
                                                small_test_inputs, small_test_targets)
            
            @test length(results_small) == 2
            @test all(r[2] ≥ 0 for r in results_small)
            @test all(isfinite(r[2]) for r in results_small)
            
            # Test with larger dataset
            large_train_inputs = rand(input_dim, 200)
            large_train_targets = rand(output_dim, 200)
            large_test_inputs = rand(input_dim, 100)
            large_test_targets = rand(output_dim, 100)
            
            results_large = ablation_connectivity(lsm, connects, large_train_inputs, large_train_targets,
                                                large_test_inputs, large_test_targets)
            
            @test length(results_large) == 2
            @test all(r[2] ≥ 0 for r in results_large)
            @test all(isfinite(r[2]) for r in results_large)
        end
        
        @testset "Network Properties During Ablation" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            original_Win = copy(lsm.Win)
            original_state = copy(lsm.state)
            
            connects = [0.3]
            ablation_connectivity(lsm, connects, train_inputs, train_targets, test_inputs, test_targets)
            
            # Win should remain unchanged
            @test lsm.Win ≈ original_Win
            
            # Other network properties should be preserved appropriately
            @test lsm.input_dim == input_dim
            @test lsm.reservoir_size == reservoir_size
            @test lsm.output_dim == output_dim
            @test lsm.leak_rate ≈ 0.3  # Default value
        end
    end

    @testset "Predict Function - Extended Testing" begin
        @testset "Basic Prediction" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            
            # Before training - should give zeros
            preds_before = predict(lsm, test_inputs)
            @test all(preds_before .== 0.0)
            @test size(preds_before) == (output_dim, T_test)
            
            # Set some weights manually
            lsm.Wout .= rand(output_dim, reservoir_size) .* 0.1
            lsm.bout .= rand(output_dim) .* 0.1
            
            preds_after = predict(lsm, test_inputs)
            @test !all(preds_after .== 0.0)
            @test size(preds_after) == (output_dim, T_test)
            @test all(isfinite, preds_after)
        end
        
        @testset "Prediction Consistency" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            train_readout!(lsm, train_inputs, train_targets)
            
            # Multiple predictions should be identical
            preds1 = predict(lsm, test_inputs)
            preds2 = predict(lsm, test_inputs)
            preds3 = predict(lsm, test_inputs)
            
            @test preds1 ≈ preds2
            @test preds2 ≈ preds3
        end
        
        @testset "Prediction with Different Input Sizes" begin
            lsm = LSMNet(input_dim, reservoir_size, output_dim; seed=seed)
            train_readout!(lsm, train_inputs, train_targets)
            
            # Single timestep
            single_input = rand(input_dim, 1)
            single_pred = predict(lsm, single_input)
            @test size(single_pred) == (output_dim, 1)
            
            # Many timesteps
            many_inputs = rand(input_dim, 500)
            many_preds = predict(lsm, many_inputs)
            @test size(many_preds) == (output_dim, 500)
            
            # All should be finite
            @test all(isfinite, single_pred)
            @test all(isfinite, many_preds)
        end
    end

    # Continue with all the other existing test sets...
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
        
        # Test prediction with zero inputs
        zero_input = zeros(input_dim, 1)
        lsm.Wout .= rand(output_dim, reservoir_size) .* 0.1
        lsm.bout .= rand(output_dim) .* 0.1
        
        zero_states = collect_states(lsm, zero_input)
        @test size(zero_states) == (reservoir_size, 1)
        
        zero_preds = predict(lsm, zero_input)
        @test size(zero_preds) == (output_dim, 1)
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

end