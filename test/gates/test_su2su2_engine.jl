# Faithfulness gate — the SU(2)charge × SU(2)spin (doubly-non-abelian) engine reproduces the U1U1
# spectrum (2I+1)(2S+1)-expanded, at the particle–hole symmetric point (εd = −U/2) where SU2SU2 is
# the maximal symmetry of the Anderson model. Independent answer = the abelian engine's many-body
# spectrum: each (I,S) multiplet of energy E must correspond to exactly (2I+1)(2S+1) U1U1 states at
# the same E. Pins the whole doubly-non-abelian engine — the doubled-6j hopping recoupling in BOTH
# sectors AND the grounded reduced-ME operator update, across the alternating isospin staggering.
# Ref: the multiplet/Wigner-Eckart NRG (Bulla–Costi–Pruschke, RMP 80, 395 (2008), §II.B).

using WilsonNRG, Test

@testset "method-recovery gate · SU2SU2 engine == U1U1 spectrum (2I+1)(2S+1)-expanded" begin
    Γ = 0.05
    D = 1.0
    disc = WilsonLog(2.5)
    keepall = KeepN(10^9)                          # no truncation: exact spectrum compare
    # block_levels(::SU2SU2) stores the FULL degeneracy (2I+1)(2S+1) as the 2nd field (not 2S).
    expand(r) = sort([round(e; digits=6) for (e, deg) in r.levels[end] for _ in 1:deg])
    plainE(r) = sort([round(e; digits=6) for (e, _) in r.levels[end]])

    # ---- (1) symmetric-point spectrum: SU2SU2 == U1U1, (I,S)-multiplet-expanded, several U ----
    for U in (0.5, 0.8, 1.3), nsites in (2, 3)
        m = AndersonModel(; U, εd=(-U / 2), Γ, D)     # p–h symmetric ⇒ SU2SU2 exact
        rsu = nrg_solve(
            m,
            NRGAlgorithm(;
                discretization=disc, symmetry=SU2SU2(), truncation=keepall, nsites
            ),
        )
        ru1 = nrg_solve(
            m,
            NRGAlgorithm(;
                discretization=disc, symmetry=U1U1(), truncation=keepall, nsites
            ),
        )
        @test length(expand(rsu)) == length(plainE(ru1))     # state counts agree (4^{nsites+1})
        @test expand(rsu) == plainE(ru1)                      # spectra identical, multiplet-expanded
    end

    # ---- (2) genuinely doubly-non-abelian: multiplets compress the state count ----
    r = nrg_solve(
        AndersonModel(; U=0.8, εd=-0.4, Γ, D),
        NRGAlgorithm(;
            discretization=disc, symmetry=SU2SU2(), truncation=keepall, nsites=3
        ),
    )
    nmult = length(r.levels[end])
    nstate = sum(deg for (_, deg) in r.levels[end])
    @test nstate > nmult                                      # (2I+1)(2S+1) members stored once

    # ---- (3) U=0 free-fermion sanity (εd = −U/2 = 0 is symmetric) ----
    m0 = AndersonModel(; U=0.0, εd=0.0, Γ, D)
    r0s = nrg_solve(
        m0,
        NRGAlgorithm(;
            discretization=disc, symmetry=SU2SU2(), truncation=keepall, nsites=2
        ),
    )
    r0u = nrg_solve(
        m0,
        NRGAlgorithm(; discretization=disc, symmetry=U1U1(), truncation=keepall, nsites=2),
    )
    @test expand(r0s) == plainE(r0u)

    # ---- (4) reduced-ME blocks connect only triangle-allowed multiplets: |I−I′|=|S−S′|=½ ----
    m = AndersonModel(; U=0.5, εd=-0.25, Γ, D)
    chain = wilson_chain(disc, m, 4)
    st = WilsonNRG.impurity_init(m, SU2SU2(), chain)
    for n in 0:3
        coupling = n == 0 ? WilsonNRG.bath_coupling(m) : chain.hopping[n]
        enl = WilsonNRG.add_site(
            st,
            SU2SU2();
            coupling,
            rescale=(n == 0 ? 1.0 : sqrt(2.5)),
            onsite=chain.onsite[n + 1],
        )
        diag = WilsonNRG.diagonalize_blocks(enl, SU2SU2())
        plan = WilsonNRG.truncation_plan(diag.vals, keepall, SU2SU2())
        st = WilsonNRG.update_operators(diag, plan, SU2SU2())
        @test all(
            abs(I - Ip) == 1 // 2 && abs(S - Sp) == 1 // 2 for (I, S, Ip, Sp) in keys(st.F)
        )
    end

    # ---- (5) truncated long chain reaches a frozen RG fixed point (multiplicity-weighted KeepN
    #          never splits a multiplet — else the flow would not converge) ----
    rfp = nrg_solve(
        AndersonModel(; U=0.5, εd=-0.25, Γ, D),
        NRGAlgorithm(;
            discretization=disc, symmetry=SU2SU2(), truncation=KeepN(60), nsites=30
        ),
    )
    lo(i) = sort(rfp.energies[i])[1:min(5, end)]
    L = lastindex(rfp.energies)
    @test maximum(abs, lo(L) .- lo(L - 2)) < 0.05             # same-parity shells frozen (fixed point)

    # ---- (6) SU2SU2 needs the p–h symmetric point; off it, honest refusal ----
    @test_throws EngineUnimplemented nrg_solve(
        AndersonModel(; U=0.4, εd=-0.3, Γ, D),
        NRGAlgorithm(; discretization=disc, symmetry=SU2SU2(), nsites=3),
    )
end
