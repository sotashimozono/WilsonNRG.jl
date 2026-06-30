# Faithfulness gate — SU(2) impurity thermodynamics. The multiplet-aware shell sum (weight 2S+1,
# ⟨Sz²⟩=S(S+1)/3) must reproduce the KNOWN free-spin / Kondo results THROUGH the non-abelian path,
# and agree with the U1U1 result shell-by-shell. This is the test that the SU(2) symmetry is
# genuinely working: a known thermodynamic answer, obtained via multiplets rather than Sz states.
# Refs: Krishna-murthy, Wilkins & Wilson 1980; Bulla–Costi–Pruschke, RMP 80, 395 (2008).

using WilsonNRG, Test

@testset "method-recovery gate · SU(2) thermodynamics (known results through multiplets)" begin
    disc = WilsonLog(2.0)
    free = AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0)        # V₀ tiny ⇒ decoupled free spin
    su2 = NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=EnergyCut(6.5), nsites=16)
    u1 = NRGAlgorithm(; discretization=disc, symmetry=U1U1(), truncation=EnergyCut(6.5), nsites=16)

    # ---- (1) decoupled ⇒ exact free spin: Tχ_imp=1/4, S_imp=ln2, high-T S→ln4 (through U1SU2) ----
    th = thermodynamics(free, su2)
    @test isapprox(th.Tχ_imp[end], 0.25; atol=0.01)        # 1/4 — needs the (2S+1)·S(S+1)/3 weighting
    @test isapprox(th.S_imp[end], log(2); atol=0.01)       # ln2 — needs the (2S+1) degeneracy in Z
    @test th.S_imp[1] > 1.30                                # free orbital ≈ ln4 = 1.386
    @test all(<(1.0e-3), diff(th.S_imp))                    # entropy monotonically quenched on cooling

    # ---- (2) cross-symmetry agreement: identical physics via multiplets vs Sz states ----
    thu = thermodynamics(free, u1)
    @test maximum(abs, th.Tχ_imp .- thu.Tχ_imp) < 1.0e-9   # U1SU2 ≡ U1U1 (same kept states, same sum)
    @test maximum(abs, th.S_imp .- thu.S_imp) < 1.0e-9

    # ---- (3) coupled ⇒ Kondo screening: Tχ_imp → 0, S_imp → 0 (through U1SU2) ----
    kondo = AndersonModel(; U=0.15, εd=-0.075, Γ=0.03, D=1.0)
    thk = thermodynamics(kondo,
        NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=EnergyCut(6.0), nsites=18))
    @test thk.Tχ_imp[end] < 0.05                            # screened singlet
    @test thk.S_imp[end] < 0.20                             # entropy quenched
    @test maximum(thk.Tχ_imp) > 2 * thk.Tχ_imp[end]         # local moment formed, then screened

    # ---- (4) magnetization through U1SU2 (Zeeman sum over the 2S+1 members) ≡ U1U1 ----
    h = 1.0e-4
    mg = magnetization(free, su2; h)
    mgu = magnetization(free, u1; h)
    @test mg.M_imp[end] / h > 0                             # paramagnetic linear response
    @test maximum(abs, mg.M_imp .- mgu.M_imp) < 1.0e-9      # agrees with the abelian path

    # ---- (5) genuinely unwired symmetries still refuse (not silently summed) ----
    @test_throws WilsonNRG.EngineUnimplemented WilsonNRG._shell_thermo([(0.0, 1)], 1.0, SU2SU2())
end
