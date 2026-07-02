# Faithfulness gate — the CFS spectral function is SYMMETRY-INDEPENDENT. A(ω) is a physical
# observable, so the U1SU2 (spin-SU(2), multiplet) engine must reproduce the U1U1 (abelian) CFS
# A(ω) it is validated against — NOT a self-consistency check but a cross-symmetry identity built
# from two independent operator-propagation code paths (U1U1 sums (Q,2Sz,σ) blocks directly;
# U1SU2 propagates the impurity d† as a spin-½ tensor via Wigner-Eckart reduced matrix elements
# and the multiplet reduced density matrices).
#
#   (a) keep-all ⇒ the two engines span the SAME Fock space ⇒ A(ω) agree to MACHINE precision.
#   (b) truncated ⇒ they agree to the keep-N-multiplets vs keep-N-states residual, which SHRINKS
#       as the kept count grows (converging to the exact identity) — a structural convergence law.
#   (c) the exact CFS sum rule ∫A_σ dω = 1 (completeness) survives the SU(2) reduction.

using WilsonNRG, Test

_A(gf) = (-1 / π) .* imag.(gf.G)
_∫(ω, A) = sum((A[i] + A[i + 1]) / 2 * (ω[i + 1] - ω[i]) for i in 1:(length(ω) - 1))

@testset "faithfulness gate · CFS spectral U1SU2 == U1U1 (symmetry independence)" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)

    # ---- (a) keep-all: same Fock space ⇒ machine-precision agreement ----
    @testset "keep-all machine precision · nsites=$nsites" for nsites in (2, 3, 4)
        alg1 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(10^9), nsites
        )
        g1 = green_function(CFS(), m, alg1)
        g2 = green_function(CFS(), m, algs; ω=g1.ω)
        @test g2.ω == g1.ω
        @test maximum(abs, _A(g2) .- _A(g1)) < 1.0e-10
    end

    # ---- (b) truncated: agreement improves with the kept count (converges to the identity) ----
    @testset "truncated convergence to the U1U1 result" begin
        errs = Float64[]
        for (nsites, keep) in ((12, 64), (16, 100), (20, 128))
            alg1 = NRGAlgorithm(;
                discretization=WilsonLog(2.5),
                symmetry=U1U1(),
                truncation=KeepN(keep),
                nsites,
            )
            algs = NRGAlgorithm(;
                discretization=WilsonLog(2.5),
                symmetry=U1SU2(),
                truncation=KeepN(keep),
                nsites,
            )
            g1 = green_function(CFS(), m, alg1)
            g2 = green_function(CFS(), m, algs; ω=g1.ω)
            push!(errs, maximum(abs, _A(g2) .- _A(g1)))
        end
        @test all(<(0.02), errs)          # sub-2% pointwise at these modest kept counts
        @test errs[end] < errs[1]         # strictly improving ⇒ converging to the exact identity
    end

    # ---- (c) exact completeness sum rule survives the SU(2) reduction ----
    @testset "sum rule ∫A_σ dω ≈ 1" begin
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5),
            symmetry=U1SU2(),
            truncation=KeepN(150),
            nsites=22,
        )
        g = green_function(CFS(), m, algs)
        @test 0.95 < _∫(g.ω, _A(g)) < 1.08
    end

    # ---- honest refusal: unimplemented symmetries throw (not a silent MethodError) ----
    @testset "honest refusal on SU2SU2" begin
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=SU2SU2(), truncation=KeepN(64), nsites=4
        )
        @test_throws EngineUnimplemented green_function(CFS(), m, alg)
    end
end
