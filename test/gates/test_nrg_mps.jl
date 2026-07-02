# Faithfulness gate — the NRG flow IS a left-canonical matrix product state (Saberi, Weichselbaum &
# von Delft, PRB 78, 035124 (2008)). The kept-eigenvector isometries A^{[n]} : (kept_{n-1} ⊗ site) →
# kept_n, contracted forward, reconstruct the NRG eigenstates as full-Fock-space vectors. At keep-all
# this reconstruction is EXACT, and it must be:
#   (a) LEFT-CANONICAL — the reconstructed states are orthonormal, `Φ'Φ = I` to machine precision
#       (the defining MPS property; a wrong isometry contraction breaks it — not a self-check);
#   (b) the ground state lives in the half-filling sector (charge Q = N_orb, spin |2Sz| = N_orb mod 2,
#       Krishna-murthy–Wilkins–Wilson even/odd alternation) with unit norm;
#   (c) a PHYSICAL observable of the reconstructed ground state matches its independent value —
#       the impurity occupation ⟨n_d⟩ = 1 exactly at the symmetric point εd = −U/2.

using WilsonNRG, Test
using LinearAlgebra: norm, dot, eigen, Symmetric

@testset "faithfulness gate · NRG is a left-canonical MPS (Saberi 2008)" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)   # symmetric point

    @testset "keep-all reconstruction · nsites=$nsites" for nsites in (2, 3, 4)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        mps = nrg_mps(m, alg)
        rec = reconstruct_mps(mps; nsites)
        Φ = rec.states

        # (a) left-canonical: the reconstructed states are an orthonormal Fock-space basis
        @test size(Φ, 1) == 4^(nsites + 1)
        @test size(Φ, 2) == 4^(nsites + 1)                 # keep-all ⇒ complete basis
        G = transpose(Φ) * Φ
        @test maximum(abs, G - one(G)) < 1.0e-10

        # (b) ground state: half-filling Q = N_orb, |2Sz| = N_orb mod 2, unit norm
        N = nsites + 1
        @test rec.gqn[1] == N
        @test abs(rec.gqn[2]) == N % 2
        gcol = findfirst(==(rec.gqn) ∘ first, rec.tags)    # any column in the ground block is unit-norm
        @test isapprox(norm(Φ[:, gcol]), 1.0; atol=1.0e-10)

        # (c) physical: reconstructed-ground impurity occupation ⟨n_d⟩ = 1 at the symmetric point
        @test isapprox(rec.gnd, 1.0; atol=1.0e-9)
    end

    # ---- Stage 2: NRG's single forward sweep is NOT the variationally-optimal MPS — vDMRG beats it
    #      at equal bond dimension, and both converge to the exact ground energy (Saberi 2008). The
    #      reference is the full Wilson-chain Hamiltonian built INDEPENDENTLY (Jordan–Wigner). ----
    @testset "vDMRG beats NRG at equal bond dimension (Saberi variational half)" begin
        Λ = 2.5
        nsites = 4
        L = nsites + 1
        allalg = NRGAlgorithm(;
            discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        H = wilson_chain_hamiltonian(m, allalg)
        F = eigen(Symmetric(H))
        Eexact = F.values[1]
        ψexact = F.vectors[:, 1]

        # (independent-H check) the NRG keep-all ground state IS H's exact ground eigenvector —
        # a fully engine-independent validation of the whole flow (and of reconstruct_mps).
        reca = reconstruct_mps(nrg_mps(m, allalg); nsites)
        ca = findfirst(==(reca.gqn) ∘ first, reca.tags)
        ΨA = reca.states[:, ca] ./ norm(reca.states[:, ca])
        @test isapprox(dot(ΨA, H * ΨA), Eexact; atol=1.0e-8)
        # ΨA is an exact ground EIGENVECTOR of the independent H (robust to the ground degeneracy —
        # nsites=4 ⇒ odd N_orb ⇒ a spin doublet, so an overlap with one picked eigenvector is ambiguous)
        @test norm(H * ΨA - Eexact * ΨA) < 1.0e-7

        # (variational) truncated: E_NRG(χ) > E_exact and shrinks with χ; the optimal bond-D MPS
        # (an upper bound on vDMRG) is STRICTLY below E_NRG at the same bond dimension.
        prevΔ = Inf
        for χ in (4, 8, 16)
            alg = NRGAlgorithm(;
                discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(χ), nsites
            )
            rec = reconstruct_mps(nrg_mps(m, alg); nsites)
            D = size(rec.states, 2)                        # NRG's total kept count ≈ its bond dim
            gc = findfirst(==(rec.gqn) ∘ first, rec.tags)
            ΨG = rec.states[:, gc] ./ norm(rec.states[:, gc])
            E_NRG = dot(ΨG, H * ΨG)
            E_opt = best_mps_energy(ψexact, H, L; D)
            @test E_NRG > Eexact - 1.0e-9                  # variational: NRG ≥ exact
            @test E_opt ≥ Eexact - 1.0e-9                  # vDMRG target ≥ exact
            @test E_opt < E_NRG - 1.0e-6                   # vDMRG STRICTLY beats NRG at bond D
            @test (E_NRG - Eexact) < prevΔ + 1.0e-12       # NRG converges as χ grows
            prevΔ = E_NRG - Eexact
        end
    end

    # honest refusal on a non-U1U1 symmetry (not a silent MethodError)
    @testset "honest refusal on U1SU2" begin
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(64), nsites=4
        )
        @test_throws EngineUnimplemented nrg_mps(m, alg)
    end
end
