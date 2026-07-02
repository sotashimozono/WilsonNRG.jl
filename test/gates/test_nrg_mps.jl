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
using LinearAlgebra: norm

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

    # honest refusal on a non-U1U1 symmetry (not a silent MethodError)
    @testset "honest refusal on U1SU2" begin
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(64), nsites=4
        )
        @test_throws EngineUnimplemented nrg_mps(m, alg)
    end
end
