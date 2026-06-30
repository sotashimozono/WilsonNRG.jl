# Faithfulness gate — impurity spectral function A(ω) via BHP T=0 patching (Stage: spectral).
# At U=0 the impurity is the resonant level A(ω)=(Γ/π)/(ω²+Γ²) (bootstrap reference). The
# log-Gaussian broadening smears the exact pointwise shape (the standard NRG artifact, sharpened
# by z-averaging — later), so the gate checks the ROBUST, sum-rule-level properties:
#  (1) spectral sum rule ∫A dω = 1 (per spin) — d† propagation + windowing are correct;
#  (2) particle–hole symmetry A(ω)=A(−ω) at the symmetric point;
#  (3) the weight is a resonance at the hybridization scale ~Γ, decaying in the band wings.
# Refs: Bulla–Hewson–Pruschke, PRB 57, 10287 (1998); Bulla–Costi–Pruschke, RMP 80, 395 (2008).

using WilsonNRG, Test

@testset "method-recovery gate · spectral function A(ω) [BHP, U=0]" begin
    Γ = 0.1
    m = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(300), nsites=26
    )
    res = spectral(BHP(), m, alg)
    ω, A = res.ω, res.A
    @test all(≥(0), A)                                          # spectral function is non-negative

    # ---- (1) sum rule ∫A dω = 1 (per spin); trapezoid on the ± log grid ----
    ∫A = sum((A[i] + A[i + 1]) / 2 * (ω[i + 1] - ω[i]) for i in 1:(length(ω) - 1))
    @test isapprox(∫A, 1.0; atol=0.12)                        # ≈1 within broadening/windowing (got ≈1.04)

    # ---- (2) particle–hole symmetry A(ω) = A(−ω) at the symmetric point ----
    npos = length(ω) ÷ 2
    Apos = A[(npos + 1):end]
    Aneg = reverse(A[1:npos])
    @test maximum(abs, Apos .- Aneg) < 1.0e-3 * maximum(A)      # exact (got ~1e-6)

    # ---- (3) resonance at the hybridization scale ~Γ, decaying into the band wings ----
    A_at(x) = A[argmin(abs.(ω .- x))]
    @test A_at(Γ) > 1.0                                         # substantial weight at ω~Γ
    @test A_at(Γ) > 5 * A_at(1.0)                               # ≫ band-edge weight (resonance, not flat)
end
