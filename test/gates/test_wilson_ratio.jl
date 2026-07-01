# Faithfulness gate — the Wilson ratio R_W → 2 in the Kondo limit (Andrei, PRL 45, 379 (1980),
# Bethe-ansatz exact; Krishna-murthy–Wilkins–Wilson, PRB 21, 1044 (1980); Hewson 1993). Independent
# answer = the UNIVERSAL strong-coupling value R=2, and R=1 for the non-interacting resonant level.
# Method (grounded, not fitted): R_W = (χ_imp/γ_imp)/(χ_imp/γ_imp)|_free with γ_imp = S_imp/T (the
# Fermi-liquid entropy slope — the fluctuation specific heat is swamped by the two-run parity
# artefact); the deep-low-T two-run breakdown cancels in the ratio (shared chain), and R_fp is the
# median over the Fermi-liquid plateau (s_lo < S_imp < s_hi). At Λ=2.5 the discretised fixed-point
# value is ≈2.0–2.2 (exact 2 as Λ→1 / z-averaging), so the gate brackets it honestly.

using WilsonNRG, Test

@testset "method-recovery gate · Wilson ratio R=2 (Andrei 1980; KWW 1980)" begin
    Γ = 0.05
    D = 1.0
    disc = WilsonLog(2.5)
    nsites = 55                                        # long enough to reach the Fermi-liquid plateau
    alg = NRGAlgorithm(;
        discretization=disc, symmetry=U1U1(), truncation=KeepN(400), nsites
    )
    RW(U) = wilson_ratio(AndersonModel(; U, εd=(-U / 2), Γ, D), alg).R_fp   # symmetric ⇒ Kondo

    # ---- (1) free resonant level (U=0): R_W = 1 exactly (the normalisation) ----
    @test wilson_ratio(AndersonModel(; U=0.0, εd=0.0, Γ, D), alg).R_fp ≈ 1.0 atol = 1.0e-9

    # ---- (2) Kondo limit: R_W ≈ 2 (U/Γ = 6, 8) and roughly universal (U/Γ-independent) ----
    r6 = RW(0.3)                                        # U/Γ = 6
    r8 = RW(0.4)                                        # U/Γ = 8
    @test 1.6 ≤ r6 ≤ 2.5                               # R_W ≈ 2 (Λ=2.5 discretised value)
    @test 1.6 ≤ r8 ≤ 2.5
    @test abs(r6 - r8) < 0.4                            # universal at the strong-coupling fixed point

    # ---- (3) crossover 1 → 2 as U/Γ grows (Kondo enhancement of χ over γ) ----
    r2 = RW(0.1)                                        # U/Γ = 2 (weak)
    @test 1.0 < r2 < r6                                 # monotone: free(1) < weak < strong(≈2)

    # ---- (4) honest stub for the not-yet-wired non-abelian thermodynamics ----
    alg2 = NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), nsites)
    @test_throws EngineUnimplemented wilson_ratio(
        AndersonModel(; U=0.4, εd=-0.2, Γ, D), alg2
    )
end
