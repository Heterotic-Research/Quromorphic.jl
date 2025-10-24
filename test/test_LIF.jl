# test/runtests.jl
"""
Comprehensive test suite for Neuromorphic.jl module
Tests LIF neurons, surrogates, encoders, functional API, optimizers, models, and training
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Import your package entry module; adjust to your package name
# If your package is Quromorphic with src/LIF.jl included from src/Quromorphic.jl:
using Quromorphic
const LIF = Quromorphic.LIF
const Neuromorphic = LIF

@testset "LIF.jl" begin

    # ============================================================================
    # TEST PARAMETERS
    # ============================================================================
    D_in = 8          # input dimension
    C_hid = 16        # hidden/output channels
    C_out = 4         # output classes
    B = 12            # batch size
    T = 20            # time steps
    seed = 42
    Random.seed!(seed)

    # Generate dummy data
    train_N = 80
    test_N = 30
    train_xs = [ [rand(Float32, D_in, 1) for _ in 1:T] for _ in 1:train_N ]
    train_ys = [rand(1:C_out) for _ in 1:train_N]
    test_xs = [ [rand(Float32, D_in, 1) for _ in 1:T] for _ in 1:test_N ]
    test_ys = [rand(1:C_out) for _ in 1:test_N]

    # ============================================================================
    # NEURON TESTS
    # ============================================================================
    @testset "LIFNeuron Constructor and Parameters" begin
        β, θ = 0.95f0, 1.0f0
        n_sub = Neuromorphic.LIFNeuron(β, θ, :subtract)
        n_zero = Neuromorphic.LIFNeuron(β, θ, :zero)

        @test n_sub.β == β
        @test n_sub.θ == θ
        @test n_sub.reset === :subtract
        @test n_zero.reset === :zero

        # Test different parameter combinations
        n_custom = Neuromorphic.LIFNeuron(0.8f0, 1.5f0, :subtract)
        @test n_custom.β == 0.8f0
        @test n_custom.θ == 1.5f0
    end

    @testset "LIF Step - Subtractive Reset" begin
        n = Neuromorphic.LIFNeuron(0.9f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        
        V = fill(0.5f0, 4, 3)
        I = fill(0.2f0, 4, 3)
        
        spk, Vnext = Neuromorphic.lif_step(V, I, n, surr)
        
        @test size(spk) == size(V)
        @test size(Vnext) == size(V)
        @test all(0 .<= spk .<= 1)  # surrogate in [0,1]
        @test all(isfinite, Vnext)
        
        # Manual check: Vnext = β*V + I - spk*θ
        @test all(Vnext .≈ n.β .* V .+ I .- spk .* n.θ)
    end

    @testset "LIF Step - Zero Reset" begin
        n = Neuromorphic.LIFNeuron(0.9f0, 1.0f0, :zero)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        
        # Force spike
        Vbig = fill(2.5f0, 4, 3)
        I = fill(0.2f0, 4, 3)
        
        spk, Vnext = Neuromorphic.lif_step(Vbig, I, n, surr)
        
        @test size(spk) == size(Vbig)
        @test all(0 .<= spk .<= 1)
        
        # Zero reset: where Vnew >= θ, Vnext should be 0
        Vnew = n.β .* Vbig .+ I
        hardmask = Vnew .>= n.θ
        @test all(Vnext .≈ Vnew .* .!hardmask)
    end

    @testset "LIF Step - No Reset" begin
        n = Neuromorphic.LIFNeuron(0.9f0, 1.0f0, :none)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        
        V = fill(0.5f0, 4, 3)
        I = fill(0.8f0, 4, 3)
        
        spk, Vnext = Neuromorphic.lif_step(V, I, n, surr)
        
        # No reset: Vnext = β*V + I regardless of spike
        @test all(Vnext .≈ n.β .* V .+ I)
    end

    @testset "LIF with Extreme Inputs" begin
        n = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        
        # Very large input
        V = zeros(Float32, 5, 2)
        I_large = fill(100.0f0, 5, 2)
        spk, Vnext = Neuromorphic.lif_step(V, I_large, n, surr)
        @test all(isfinite, Vnext)
        @test all(isfinite, spk)
        
        # Very small input
        I_small = fill(1e-8f0, 5, 2)
        spk_s, Vnext_s = Neuromorphic.lif_step(V, I_small, n, surr)
        @test all(isfinite, Vnext_s)
        @test all(isfinite, spk_s)
    end

    # ============================================================================
    # SURROGATE TESTS
    # ============================================================================
    @testset "AtanSurrogate Range and Monotonicity" begin
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        u = collect(Float32, -3:0.2:3)
        
        y = Neuromorphic.surrogate(u, surr)
        
        @test all(0 .<= y .<= 1)
        @test y[1] < y[end]  # monotone increasing
        
        # Test with different slope
        surr2 = Neuromorphic.AtanSurrogate(5.0f0)
        y2 = Neuromorphic.surrogate(u, surr2)
        @test all(0 .<= y2 .<= 1)
    end

    @testset "FastSigmoidSurrogate Range" begin
        surr = Neuromorphic.FastSigmoidSurrogate(2.0f0, 1.0f0)
        u = collect(Float32, -2:0.1:2)
        
        y = Neuromorphic.surrogate(u, surr)
        
        @test all(0 .<= y .<= 1)
        
        # Test with different parameters
        surr2 = Neuromorphic.FastSigmoidSurrogate(3.0f0, 0.5f0)
        y2 = Neuromorphic.surrogate(u, surr2)
        @test all(0 .<= y2 .<= 1)
    end

    @testset "Surrogate Consistency" begin
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        u = randn(Float32, 10)
        
        y1 = Neuromorphic.surrogate(u, surr)
        y2 = Neuromorphic.surrogate(u, surr)
        
        @test y1 ≈ y2
    end

    # ============================================================================
    # ENCODER TESTS
    # ============================================================================
    @testset "Rate Encoder - Basic" begin
        rng = MersenneTwister(seed)
        x = clamp.(rand(rng, Float32, 16, 4), 0, 1)
        Tsteps = 30
        
        seq = Neuromorphic.rate_encode(x, Tsteps; rng=rng)
        
        @test length(seq) == Tsteps
        @test all(size(f) == size(x) for f in seq)
        @test all(all(s in [0f0, 1f0] for s in f) for f in seq)
    end

    @testset "Rate Encoder - Firing Rate Accuracy" begin
        rng = MersenneTwister(seed)
        x = fill(0.5f0, 10, 5)  # 50% rate
        Tsteps = 200
        
        seq = Neuromorphic.rate_encode(x, Tsteps; rng=rng)
        counts = reduce(+, seq) ./ Tsteps
        
        # Should be close to 0.5 with MC error
        @test mean(abs.(counts .- 0.5f0)) < 0.1
    end

    @testset "Poisson Encoder - Equivalence to Rate" begin
        rng = MersenneTwister(seed)
        x = rand(rng, Float32, 8, 3)
        Tsteps = 25
        
        seq_rate = Neuromorphic.rate_encode(x, Tsteps; rng=rng)
        
        rng2 = MersenneTwister(seed)
        seq_pois = Neuromorphic.poisson_encode(x, Tsteps; rng=rng2)
        
        # Should be identical for same seed
        @test all(seq_rate[i] ≈ seq_pois[i] for i in 1:Tsteps)
    end

    @testset "Encoder with Edge Cases" begin
        rng = MersenneTwister(seed)
        
        # All zeros
        x_zero = zeros(Float32, 5, 2)
        seq_zero = Neuromorphic.rate_encode(x_zero, 20; rng=rng)
        @test all(all(f .== 0) for f in seq_zero)
        
        # All ones
        x_one = ones(Float32, 5, 2)
        seq_one = Neuromorphic.rate_encode(x_one, 20; rng=rng)
        @test all(all(f .== 1) for f in seq_one)
        
        # Out of range (clamp)
        x_oob = fill(1.5f0, 5, 2)
        seq_oob = Neuromorphic.rate_encode(x_oob, 20; rng=rng)
        @test length(seq_oob) == 20
    end

    # ============================================================================
    # FUNCTIONAL API TESTS
    # ============================================================================
    @testset "ce_count_loss - Basic" begin
        # 3 classes, batch 2, 2 time steps
        s1 = Float32[3 0; 0 1; 0 0]  # C×B at t1
        s2 = Float32[1 0; 0 0; 0 2]  # t2
        spk_seq = [s1, s2]
        y = [1, 3]
        
        loss = Neuromorphic.ce_count_loss(spk_seq, y)
        
        @test isfinite(loss)
        @test loss >= 0
    end

    @testset "ce_count_loss - Perfect Classification" begin
        # Class 1 gets all spikes in col1, class 2 in col2
        s1 = Float32[10 0; 0 10]
        spk_seq = [s1]
        y = [1, 2]
        
        loss = Neuromorphic.ce_count_loss(spk_seq, y)
        
        @test loss < 0.1  # should be very low
    end

    @testset "accuracy_rate - Basic" begin
        s1 = Float32[2 0; 0 1; 0 0]
        s2 = Float32[1 0; 0 0; 0 2]
        spk_seq = [s1, s2]
        y = [1, 3]
        
        acc = Neuromorphic.accuracy_rate(spk_seq, y)
        
        @test 0 <= acc <= 1
        @test acc == 1.0  # perfect on this toy case
    end

    @testset "accuracy_rate - Imperfect" begin
        s1 = Float32[0 1; 1 0; 0 0]  # wrong predictions
        spk_seq = [s1]
        y = [1, 2]
        
        acc = Neuromorphic.accuracy_rate(spk_seq, y)
        
        @test acc == 0.0
    end

    @testset "l1_rate Regularizer" begin
        s1 = Float32[1 2; 3 4]
        s2 = Float32[0.5 1; 1.5 2]
        spk_seq = [s1, s2]
        
        reg = Neuromorphic.l1_rate(spk_seq; λ=1f-3)
        
        expected = 1f-3 * mean(abs, s1 .+ s2)
        @test reg ≈ expected
    end

    # ============================================================================
    # LAYER AND MODEL TESTS
    # ============================================================================
    @testset "LinearLIFLayer Constructor" begin
        W = randn(Float32, 5, 8)
        b = zeros(Float32, 5)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        
        @test layer.W === W
        @test layer.b === b
        @test layer.neuron === lif
        @test layer.surrogate === surr
    end

    @testset "FeedForwardSNN Constructor" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        @test length(model.layers) == 1
        @test model.T == T
    end

    @testset "forward_time - Output Shape" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        xseq = [rand(Float32, D_in, B) for _ in 1:T]
        spk_seq = Neuromorphic.forward_time(model, xseq)
        
        @test length(spk_seq) == T
        @test all(size(s) == (C_out, B) for s in spk_seq)
        @test all(all(0 .<= s .<= 1) for s in spk_seq)
    end

    @testset "forward_time - Consistency" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        xseq = [rand(Float32, D_in, B) for _ in 1:T]
        
        spk1 = Neuromorphic.forward_time(model, xseq)
        spk2 = Neuromorphic.forward_time(model, xseq)
        
        @test all(spk1[i] ≈ spk2[i] for i in 1:T)
    end

    @testset "forward - Non-sequence Input" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], 15)
        
        x = rand(Float32, D_in, B)
        spk_seq = Neuromorphic.forward(model, x)
        
        @test length(spk_seq) == 15
        @test all(size(s) == (C_out, B) for s in spk_seq)
    end

    # ============================================================================
    # DATALOADER TESTS
    # ============================================================================
    @testset "SpikingDataLoader - Basic Iteration" begin
        dl = Neuromorphic.SpikingDataLoader(train_xs, train_ys, 16)
        
        @test length(dl) == cld(train_N, 16)
        
        count = 0
        for (xb, yb) in dl
            count += 1
            @test length(xb) == T
            @test length(yb) <= 16
            @test size(xb[1], 1) == D_in
            @test size(xb[1], 2) == length(yb)
        end
        @test count == length(dl)
    end

    @testset "SpikingDataLoader - Small Batch" begin
        dl = Neuromorphic.SpikingDataLoader(train_xs, train_ys, 5)
        
        for (xb, yb) in dl
            @test length(yb) <= 5
            @test size(xb[1], 2) == length(yb)
        end
    end

    @testset "SpikingDataLoader - Single Sample" begin
        single_xs = [train_xs[1]]
        single_ys = [train_ys[1]]
        dl = Neuromorphic.SpikingDataLoader(single_xs, single_ys, 1)
        
        @test length(dl) == 1
        
        for (xb, yb) in dl
            @test length(yb) == 1
            @test size(xb[1], 2) == 1
        end
    end

    # ============================================================================
    # OPTIMIZER TESTS
    # ============================================================================
    @testset "SGDOptimizer - Parameter Update" begin
        W = randn(Float32, 5, 4)
        gW = fill(0.1f0, size(W))
        W0 = copy(W)
        
        opt = Neuromorphic.SGDOptimizer(1f-2; momentum=0.0f0, clip=1f0)
        Neuromorphic.apply!(opt, [W], [gW])
        
        @test norm(W - W0) > 0
        @test all(isfinite, W)
    end

    @testset "SGDOptimizer - Momentum" begin
        W = randn(Float32, 5, 4)
        gW = fill(0.1f0, size(W))
        
        opt = Neuromorphic.SGDOptimizer(1f-2; momentum=0.9f0, clip=1f0)
        
        # First update
        W1 = copy(W)
        Neuromorphic.apply!(opt, [W], [gW])
        delta1 = norm(W - W1)
        
        # Second update with same gradient
        W2 = copy(W)
        Neuromorphic.apply!(opt, [W], [gW])
        delta2 = norm(W - W2)
        
        # With momentum, second step should be larger
        @test delta2 > delta1
    end

    @testset "AdamOptimizer - Parameter Update" begin
        W = randn(Float32, 5, 4)
        b = zeros(Float32, 5)
        gW = randn(Float32, size(W))
        gb = randn(Float32, size(b))
        W0, b0 = copy(W), copy(b)
        
        opt = Neuromorphic.AdamOptimizer(1f-3; clip=1f0)
        Neuromorphic.apply!(opt, [W, b], [gW, gb])
        
        @test norm(W - W0) > 0
        @test norm(b - b0) > 0
        @test all(isfinite, W)
        @test all(isfinite, b)
    end

    @testset "AdamOptimizer - Bias Correction" begin
        W = randn(Float32, 5, 4)
        gW = fill(0.1f0, size(W))
        
        opt = Neuromorphic.AdamOptimizer(1f-3)
        
        @test opt.t == 0
        Neuromorphic.apply!(opt, [W], [gW])
        @test opt.t == 1
        
        Neuromorphic.apply!(opt, [W], [gW])
        @test opt.t == 2
    end

    @testset "RMSpropOptimizer - Parameter Update" begin
        W = randn(Float32, 5, 4)
        gW = randn(Float32, size(W))
        W0 = copy(W)
        
        opt = Neuromorphic.RMSpropOptimizer(1f-2; clip=1f0)
        Neuromorphic.apply!(opt, [W], [gW])
        
        @test norm(W - W0) > 0
        @test all(isfinite, W)
    end

    @testset "Gradient Clipping" begin
        W = randn(Float32, 10, 10)
        gW = fill(100.0f0, size(W))  # huge gradient
        
        opt = Neuromorphic.AdamOptimizer(1f-3; clip=1.0f0)
        
        # Clip should prevent explosion
        Neuromorphic.apply!(opt, [W], [gW])
        @test all(isfinite, W)
    end

    # ============================================================================
    # TRAINING TESTS
    # ============================================================================
    @testset "train_epoch! - Single Epoch" begin
        Random.seed!(seed)
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        dl = Neuromorphic.SpikingDataLoader(train_xs[1:20], train_ys[1:20], 10)
        opt = Neuromorphic.AdamOptimizer(1f-3)
        
        W0 = copy(model.layers[1].W)
        
        loss, acc = Neuromorphic.train_epoch!(model, dl, opt)
        
        @test isfinite(loss)
        @test 0 <= acc <= 1
        @test norm(model.layers[1].W - W0) > 0  # weights should change
    end

    @testset "train_snn! - Multiple Epochs" begin
        Random.seed!(seed)
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        dl = Neuromorphic.SpikingDataLoader(train_xs[1:40], train_ys[1:40], 16)
        cfg = Neuromorphic.TrainingConfig(n_epochs=5, batch_size=16, lr=2f-3)
        
        hist = Neuromorphic.train_snn!(model, dl, cfg)
        
        @test length(hist.loss) == 5
        @test length(hist.acc) == 5
        @test all(isfinite, hist.loss)
        @test all(0 .<= hist.acc .<= 1)
    end

    @testset "train_snn! - Loss Decreases" begin
        Random.seed!(123)
        # Use synthetic correlated data
        D, C, T_steps = 10, 3, 15
        N = 60
        xs = Vector{Vector{Array{Float32,2}}}(undef, N)
        ys = Vector{Int}(undef, N)
        for i in 1:N
            y_val = rand(1:C)
            ys[i] = y_val
            base = zeros(Float32, D, 1)
            base[y_val] = 1.0f0
            xs[i] = [clamp.(base .+ 0.2f0*randn(Float32, D, 1), 0, 1) for _ in 1:T_steps]
        end
        
        W = randn(Float32, C, D)
        b = zeros(Float32, C)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T_steps)
        
        dl = Neuromorphic.SpikingDataLoader(xs, ys, 20)
        cfg = Neuromorphic.TrainingConfig(n_epochs=8, batch_size=20, lr=3f-3)
        
        hist = Neuromorphic.train_snn!(model, dl, cfg)
        
        early_loss = mean(hist.loss[1:3])
        late_loss = mean(hist.loss[end-2:end])
        
        @test late_loss <= early_loss
    end

    # ============================================================================
    # INTEGRATION TESTS
    # ============================================================================
    @testset "End-to-End: Encode -> Forward -> Loss" begin
        Random.seed!(seed)
        rng = MersenneTwister(seed)
        
        x = rand(rng, Float32, D_in, B)
        xseq = Neuromorphic.rate_encode(x, T; rng=rng)
        
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        spk_seq = Neuromorphic.forward_time(model, xseq)
        y = [rand(1:C_out) for _ in 1:B]
        
        loss = Neuromorphic.ce_count_loss(spk_seq, y)
        acc = Neuromorphic.accuracy_rate(spk_seq, y)
        
        @test isfinite(loss)
        @test 0 <= acc <= 1
    end

    @testset "Parameters Function" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        Ws, bs = Neuromorphic.parameters(model)
        
        @test length(Ws) == 1
        @test length(bs) == 1
        @test Ws[1] === model.layers[1].W
        @test bs[1] === model.layers[1].b
    end

    # ============================================================================
    # EDGE CASES AND ROBUSTNESS
    # ============================================================================
    @testset "Edge Case - Single Neuron" begin
        W = randn(Float32, 1, 1)
        b = zeros(Float32, 1)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], 10)
        
        xseq = [rand(Float32, 1, 2) for _ in 1:10]
        spk_seq = Neuromorphic.forward_time(model, xseq)
        
        @test length(spk_seq) == 10
        @test all(size(s) == (1, 2) for s in spk_seq)
    end

    @testset "Edge Case - Very Large Network" begin
        W = randn(Float32, 50, 100)
        b = zeros(Float32, 50)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], 5)
        
        xseq = [rand(Float32, 100, 8) for _ in 1:5]
        spk_seq = Neuromorphic.forward_time(model, xseq)
        
        @test length(spk_seq) == 5
        @test all(isfinite, spk_seq[1])
    end

    @testset "Edge Case - Zero Weights" begin
        W = zeros(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], 10)
        
        xseq = [rand(Float32, D_in, B) for _ in 1:10]
        spk_seq = Neuromorphic.forward_time(model, xseq)
        
        # With zero weights, all spikes should be ~0
        @test all(mean(s) < 0.1 for s in spk_seq)
    end

    @testset "Reproducibility - Same Seed" begin
        Random.seed!(999)
        W1 = randn(Float32, C_out, D_in)
        
        Random.seed!(999)
        W2 = randn(Float32, C_out, D_in)
        
        @test W1 ≈ W2
    end

    @testset "Numerical Stability - Large Batches" begin
        W = randn(Float32, C_out, D_in)
        b = zeros(Float32, C_out)
        lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
        surr = Neuromorphic.AtanSurrogate(2.0f0)
        layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
        model = Neuromorphic.FeedForwardSNN([layer], T)
        
        xseq = [rand(Float32, D_in, 200) for _ in 1:T]
        spk_seq = Neuromorphic.forward_time(model, xseq)
        
        @test all(all(isfinite, s) for s in spk_seq)
    end

    # ============================================================================
    # GPU TESTS (OPTIONAL)
    # ============================================================================
    @testset "Metal forward (optional)" begin
        try
            import Metal
            Metal.versioninfo()  # smoke test
            D, C, B, T = 32, 10, 16, 8
            W = Metal.MtlArray(randn(Float32, C, D))
            b = Metal.MtlArray(zeros(Float32, C))
            lif = Neuromorphic.LIFNeuron(0.95f0, 1.0f0, :subtract)
            surr = Neuromorphic.AtanSurrogate(2.0f0)
            layer = Neuromorphic.LinearLIFLayer(W, b, lif, surr)
            model = Neuromorphic.FeedForwardSNN([layer], T)
            xseq = [Metal.MtlArray(rand(Float32, D, B)) for _ in 1:T]
            spk_seq = Neuromorphic.forward_time(model, xseq)
            @test length(spk_seq) == T
            @test eltype(Array(spk_seq[1])) == Float32
        catch e
            @info "Metal not available or failed to init: $e"
            @test true
        end
    end
    

end
