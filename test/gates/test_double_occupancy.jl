# Faithfulness gate — impurity double occupancy ⟨n_{d↑}n_{d↓}⟩, the charge-fluctuation static
# property of Krishna-murthy, Wilkins & Wilson II (PRB 21, 1044 (1980)). Evaluated by propagating
# the n↑n↓ observable along the flow and reading ⟨G|n↑n↓|G⟩.
#  (1) U=0 ⇒ the spins are UNCORRELATED, so ⟨n↑n↓⟩ = ⟨n↑⟩⟨n↓⟩ = n_{dσ}² EXACTLY (tier-1: the
#      per-spin occupation is the independently-checked single-particle value of test_occupation).
#  (2) U>0 ⇒ the Coulomb repulsion SUPPRESSES double occupancy below the uncorrelated value,
#      monotonically in U, → 0 as U → ∞ (the local-moment formation of the asymmetric model).
#  (3) bounds: 0 ≤ ⟨n↑n↓⟩ ≤ min(⟨n↑⟩,⟨n↓⟩); honest EngineUnimplemented stub for U1SU2.

using WilsonNRG, Test

@testset "method-recovery gate · impurity double occupancy ⟨n↑n↓⟩ (KWW II charge fluctuation)" begin
    Γ = 0.05
    D = 1.0
    Λ = 2.5
    ka(nsites) = NRGAlgorithm(;
        discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(10^9), nsites
    )

    # ---- (1) U=0 ⇒ ⟨n↑n↓⟩ = ⟨n↑⟩⟨n↓⟩ exactly (spins uncorrelated; keep-all ⇒ tier-1 exact) ----
    @testset "U=0 spin factorization ⟨n↑n↓⟩=⟨n↑⟩⟨n↓⟩" begin
        for εd in (0.0, 0.3, -0.2, 0.5)
            model = AndersonModel(; U=0.0, εd, Γ, D)
            nσ = occupation(model, ka(5)).up
            docc = double_occupancy(model, ka(5))
            @test docc ≈ nσ^2 atol = 1.0e-9
            @test 0.0 ≤ docc ≤ nσ + 1.0e-12                 # physical bound
        end
    end

    # ---- (2) U>0 ⇒ Coulomb suppression below the uncorrelated value, monotone in U ----
    @testset "Coulomb suppression at the symmetric point" begin
        # symmetric point: ⟨n↑⟩=⟨n↓⟩=1/2 ⇒ uncorrelated value 1/4; U suppresses below it, → 0
        d = [double_occupancy(AndersonModel(; U, εd=-U / 2, Γ, D), ka(6)) for U in (0.0, 0.4, 1.0, 2.0)]
        @test d[1] ≈ 0.25 atol = 1.0e-6                     # U=0 symmetric ⇒ exactly 1/4
        @test issorted(d; rev=true)                         # monotonically suppressed by U
        @test all(0.0 .≤ d .≤ 0.25 + 1.0e-9)                # never exceeds the uncorrelated value
        @test d[end] < 0.1                                  # strong U ⇒ double occ nearly frozen out
    end

    # ---- (3) truncated production run agrees with keep-all (observable is truncation-robust) ----
    @testset "truncated run matches keep-all" begin
        model = AndersonModel(; U=0.6, εd=-0.3, Γ, D)
        exact = double_occupancy(model, ka(7))
        trunc = double_occupancy(
            model,
            NRGAlgorithm(; discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(400), nsites=7),
        )
        @test trunc ≈ exact atol = 1.0e-5           # truncation-robust (→ exact as KeepN grows)
    end

    # ---- honest stub for an unwired symmetry ----
    @testset "honest stub for unwired symmetry" begin
        model = AndersonModel(; U=0.4, εd=-0.2, Γ, D)
        alg = NRGAlgorithm(; discretization=WilsonLog(Λ), symmetry=U1SU2(), nsites=5)
        @test_throws EngineUnimplemented double_occupancy(model, alg)
    end
end
