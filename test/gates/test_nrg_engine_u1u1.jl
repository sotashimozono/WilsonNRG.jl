# Faithfulness gate — iterative NRG engine, WilsonLog + U1U1 + AndersonModel (Stage 1).
# no-cite-without-reproduction checks for the iterative diagonalization:
#  (1) U=0 ⇒ the many-body spectrum is EXACTLY the free-fermion subset-sums of the
#      single-particle Wilson chain (independent diagonalization). Tests the whole
#      block-assembly + fermion-sign + √Λ-rescaling machinery. tier-1, exact.
#  (2) symmetric Anderson (εd=-U/2, flat band) has exact spin-flip and particle–hole
#      symmetry of the spectrum.
#  (3) half-filling ground state with the NRG even/odd spin alternation (Kondo singlet
#      at even site count, doublet at odd) — KWW 1980 / Bulla–Costi–Pruschke 2008.
# Scope: U(1)×U(1) abelian engine, energy flow. Spectral functions / thermodynamics
# are later stages — not claimed here.

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigvals
using WilsonNRG:
    impurity_init, add_site, diagonalize_blocks, update_operators, truncation_plan

# independent free-fermion reference: single-particle Wilson-chain matrix, same √Λ recursion
function _single_particle(model, chain, nsites)
    Λ = chain.disc.Λ
    m = reshape([model.εd], 1, 1)
    for n in 0:(nsites - 1)
        c = n == 0 ? bath_coupling(model) : chain.hopping[n]
        r = n == 0 ? 1.0 : sqrt(Λ)
        k = size(m, 1)
        mn = zeros(k + 1, k + 1)
        mn[1:k, 1:k] = r .* m
        mn[k + 1, k + 1] = chain.onsite[n + 1]
        mn[k, k + 1] = c
        mn[k + 1, k] = c
        m = mn
    end
    return eigvals(Symmetric(m))
end
function _subset_sums(levels)
    s = [0.0]
    for λ in levels
        s = vcat(s, s .+ λ)
    end
    return sort(s)
end

# run the engine keeping all states, returning the block-resolved spectrum at the last step
function _keepall_blocks(model, Λ, nsites)
    chain = wilson_chain(WilsonLog(Λ), model, nsites)
    st = impurity_init(model, U1U1(), chain)
    for n in 0:(nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrt(Λ)
        diag = diagonalize_blocks(
            add_site(st, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        st = update_operators(diag, truncation_plan(diag.vals, KeepN(10^9), U1U1()), U1U1())
    end
    return st.E
end

@testset "method-recovery gate · NRG engine (U1U1 Anderson)" begin
    # ---- (1) U=0 ⇒ exact free fermions (via the public nrg_solve API) ----
    @testset "U=0 free-fermion reproduction" begin
        for (U, εd) in ((0.0, 0.0), (0.0, 0.3))
            model = AndersonModel(; U, εd, Γ=0.05, D=1.0)
            alg = NRGAlgorithm(;
                discretization=WilsonLog(2.5),
                symmetry=U1U1(),
                truncation=KeepN(10^9),
                nsites=5,           # keep-all ⇒ exact
            )
            got = sort(nrg_solve(model, alg).energies[end])
            sp = _single_particle(model, wilson_chain(WilsonLog(2.5), model, 5), 5)
            ref = _subset_sums(vcat(sp, sp))                    # ↑ and ↓ spin channels
            @test length(got) == length(ref) == 4096
            # compare relative to ground (the engine ground-subtracts each iteration)
            @test maximum(abs, (got .- got[1]) .- (ref .- ref[1])) < 1e-9
        end
    end

    # ---- (2)+(3) symmetric Anderson: symmetries + even/odd ground state ----
    @testset "symmetric-point symmetries & even/odd flow" begin
        model = AndersonModel(; U=0.4, εd=-0.2, Γ=0.05, D=1.0)   # εd = -U/2
        for nsites in (3, 4, 5, 6)                                # both parities; keep-all stays small
            E = _keepall_blocks(model, 2.5, nsites)
            N = nsites + 1                                        # orbitals = impurity + f0..f_{nsites-1}

            # spin-flip: E[(Q,D)] == E[(Q,-D)]
            sf = maximum(
                maximum(abs, sort(E[(Q, D)]) .- sort(E[(Q, -D)])) for
                (Q, D) in keys(E) if haskey(E, (Q, -D))
            )
            @test sf < 1e-9

            # particle–hole: E[(Q,D)] == E[(2N-Q, D)]
            ph = maximum(
                maximum(abs, sort(E[(Q, D)]) .- sort(E[(2N - Q, D)])) for
                (Q, D) in keys(E) if haskey(E, (2N - Q, D))
            )
            @test ph < 1e-9

            # ground state: half-filling Q=N, spin |D| = N mod 2 (singlet/doublet alternation)
            gmin = minimum(minimum(v) for v in values(E))
            gqn = first((Q, D) for (Q, D) in keys(E) if minimum(E[(Q, D)]) ≈ gmin)
            @test gqn[1] == N
            @test abs(gqn[2]) == N % 2
        end
    end

    # ---- truncated run is stable and bounded (the production path) ----
    @testset "truncated flow" begin
        model = AndersonModel(; U=0.4, εd=-0.2, Γ=0.05, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(256), nsites=35
        )
        res = nrg_solve(model, alg)
        @test length(res.energies) == 35
        @test all(isfinite, reduce(vcat, res.energies))
        @test 256 ≤ maximum(res.kept) ≤ 320         # ≥ N_keep, extended through degeneracy (no split)
        @test all(>(0), res.kept)
        @test all(e -> minimum(e) ≈ 0, res.energies)  # ground-subtracted: each iteration starts at 0
    end

    # ---- RG flow reaches a fixed point (hallmark of NRG; also exercises the
    #      ground-subtraction numerical-stability fix on a long chain) ----
    @testset "RG flow reaches a fixed point" begin
        model = AndersonModel(; U=0.5, εd=-0.25, Γ=0.04, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(300), nsites=50
        )
        res = nrg_solve(model, alg)
        @test all(isfinite, reduce(vcat, res.energies))
        @test maximum(res.energies[end]) < 100          # rescaled spectrum O(1): no √Λ^N blow-up
        lo(n) = sort(res.energies[n])[1:6]
        # same-parity late iterations ⇒ the rescaled low-energy spectrum is frozen
        @test maximum(abs, lo(lastindex(res.energies)) .- lo(lastindex(res.energies) - 2)) <
            0.05
    end

    # ---- SU2SU2 needs the particle–hole symmetric point (εd = −U/2); off it, honest refusal ----
    @testset "honest refusal off the SU2SU2 symmetric point" begin
        alg = NRGAlgorithm(; discretization=WilsonLog(2.5), symmetry=SU2SU2(), nsites=3)
        asym = AndersonModel(; U=0.4, εd=-0.1, Γ=0.05, D=1.0)      # εd ≠ −U/2 = −0.2
        @test_throws EngineUnimplemented nrg_solve(asym, alg)
    end
end
