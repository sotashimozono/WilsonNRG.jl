# Faithfulness gate — the U1×SU2 non-abelian engine reproduces the U1U1 spectrum (2S+1)-expanded.
# This is cross-method agreement (the independent answer = the abelian engine's many-body spectrum):
# each U1SU2 multiplet (Q,S) of energy E must correspond to exactly (2S+1) U1U1 states at the same E.
# It pins the whole non-abelian engine — the 6j hopping recoupling AND the reduced-ME operator update.
# Refs: the multiplet/Wigner-Eckart NRG (Bulla–Costi–Pruschke, RMP 80, 395 (2008), §II.B).

using WilsonNRG, Test

@testset "method-recovery gate · U1SU2 engine == U1U1 spectrum (2S+1)-expanded" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.1, D=1.0)
    keepall = KeepN(10^9)                                  # no truncation: exact spectrum compare
    disc = WilsonLog(2.5)

    for nsites in (2, 3)
        alg_su2 = NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=keepall, nsites)
        alg_u1 = NRGAlgorithm(; discretization=disc, symmetry=U1U1(), truncation=keepall, nsites)
        rsu2 = nrg_solve(m, alg_su2)
        ru1 = nrg_solve(m, alg_u1)
        # expand each U1SU2 multiplet (E, 2S) into (2S+1) copies; compare the final-shell energy multiset
        esu2 = sort([round(e; digits=6) for (e, twoS) in rsu2.levels[end] for _ in 1:(twoS + 1)])
        eu1 = sort([round(e; digits=6) for (e, _) in ru1.levels[end]])
        @test length(esu2) == length(eu1)                 # state counts agree (4^{nsites+1})
        @test esu2 == eu1                                  # spectra identical, multiplet-expanded
    end

    # genuinely non-abelian: multiplets compress the state count (Σ(2S+1) > #multiplets)
    r = nrg_solve(m, NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=keepall, nsites=3))
    nmult = length(r.levels[end])
    nstate = sum(twoS + 1 for (e, twoS) in r.levels[end])
    @test nstate > nmult                                   # storing multiplets once, not 2S+1 members

    # U=0 sanity: still reproduces the free-fermion-subset spectrum (no interaction-specific bug)
    m0 = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D=1.0)
    r0s = nrg_solve(m0, NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=keepall, nsites=2))
    r0u = nrg_solve(m0, NRGAlgorithm(; discretization=disc, symmetry=U1U1(), truncation=keepall, nsites=2))
    e0s = sort([round(e; digits=6) for (e, twoS) in r0s.levels[end] for _ in 1:(twoS + 1)])
    e0u = sort([round(e; digits=6) for (e, _) in r0u.levels[end]])
    @test e0s == e0u

    # thermodynamics/magnetization must REFUSE a U1SU2 result (its levels are 2S, not 2Sz) —
    # guarded, not silently mis-summed. SU(2)-correct thermo is a separate (multiplet-aware) layer.
    alg_su2 = NRGAlgorithm(; discretization=disc, symmetry=U1SU2(), truncation=keepall, nsites=2)
    @test_throws WilsonNRG.EngineUnimplemented thermodynamics(m, alg_su2)
    @test_throws WilsonNRG.EngineUnimplemented magnetization(m, alg_su2; h=0.01)
end
