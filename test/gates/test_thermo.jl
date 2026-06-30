# Faithfulness gate — impurity thermodynamics (Stage 2).
# no-cite-without-reproduction checks for the two-run (full − bath) NRG thermodynamics
# (Krishna-murthy, Wilkins & Wilson 1980; Bulla, Costi & Pruschke, RMP 80, 395 (2008)):
#  (1) a DECOUPLED impurity is an exact free spin — its low-T contribution must be
#      T·χ_imp = 1/4 and S_imp = ln2, and the SAME run shows the free-orbital→local-moment
#      crossover (high-T entropy → ln4).  [tight, exact]
#  (2) a COUPLED symmetric impurity is Kondo-screened at low T: T·χ_imp → 0, S_imp → 0.
# Truncation MUST be EnergyCut here: a fixed KeepN under-resolves the impurity-doubled
# full run and the local-moment plateau undershoots (see src/thermodynamics.jl).
# Scope: U(1)×U(1) Anderson, entropy + susceptibility. Specific heat / spectral fns are later.

using WilsonNRG, Test

@testset "method-recovery gate · impurity thermodynamics" begin
    # ---- (1) decoupled ⇒ exact free spin + free-orbital→local-moment crossover ----
    @testset "decoupled free spin: Tχ=1/4, S=ln2 (and ln4 at high T)" begin
        th = thermodynamics(
            AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0),   # V₀ tiny ⇒ T_K→0
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.5),
                nsites=16,
            );
            betabar=1.0,
        )
        # low-T local moment (exact free spin) — the tight check
        @test isapprox(th.Tχ_imp[end], 0.25; atol=0.01)        # 1/4 (got ≈0.2487)
        @test isapprox(th.S_imp[end], log(2); atol=0.01)       # ln2 (got ≈0.6933)
        # high-T free orbital: 4 impurity states ⇒ S_imp → ln4
        @test th.S_imp[1] > 1.30                                 # ≈ ln4 = 1.386 (got ≈1.37)
        # crossover direction: cooling quenches entropy; Tχ rises 1/8-side → 1/4
        @test all(<(1.0e-3), diff(th.S_imp))                     # S_imp monotonically non-increasing
        @test th.Tχ_imp[1] < 0.18 < th.Tχ_imp[end]               # rises into the local moment
    end

    # ---- (2) coupled ⇒ Kondo screening: Tχ_imp → 0, S_imp → 0 at low T ----
    @testset "Kondo screening: Tχ_imp → 0, S_imp → 0" begin
        th = thermodynamics(
            AndersonModel(; U=0.15, εd=-0.075, Γ=0.03, D=1.0),   # U/Γ=5 ⇒ T_K reachable
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.0),
                nsites=18,
            );
            betabar=1.0,
        )
        @test th.Tχ_imp[end] < 0.05                              # screened singlet (got ≈0.027)
        @test th.S_imp[end] < 0.20                               # entropy quenched (got ≈0.135)
        @test maximum(th.Tχ_imp) > 2 * th.Tχ_imp[end]            # rose (local moment) then screened
        @test all(isfinite, th.Tχ_imp) && all(isfinite, th.S_imp)
    end
end
