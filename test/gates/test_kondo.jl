# Faithfulness gate — Kondo model on the SAME generic engine (model-dispatch, axis 1).
# Only impurity_init differs from Anderson; the recursion + flow are reused.
#  (1) init spectrum is exact: impurity-f₀ exchange gives a singlet at −3J/4 and a
#      triplet at J/4 (spin-½ ⊗ spin-½: S·s = −3/4 / +1/4), with f₀ empty/double at 0.
#  (2) J=0 ⇒ the impurity spin decouples: many-body spectrum = free conduction chain
#      × free spin (×2), EXACTLY — validates the f†₀ bookkeeping. [tight, exact]
#  (3) J>0 ⇒ the flow reaches the Kondo strong-coupling fixed point.
# Refs: Wilson, RMP 47, 773 (1975); Krishna-murthy, Wilkins & Wilson, PRB 21, 1044 (1980).

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigvals
using WilsonNRG: impurity_init

# single-particle spectrum of the bare conduction chain f₀…f_{N-1} (Kondo rescaling)
function _chain_sp(chain, nsites)
    Λ = chain.disc.Λ
    m = reshape([chain.onsite[1]], 1, 1)
    for n in 1:(nsites - 1)
        c = chain.hopping[n]
        k = size(m, 1)
        mn = zeros(k + 1, k + 1)
        mn[1:k, 1:k] = sqrt(Λ) .* m
        mn[k + 1, k + 1] = chain.onsite[n + 1]
        mn[k, k + 1] = c
        mn[k + 1, k] = c
        m = mn
    end
    return eigvals(Symmetric(m))
end
_subset(ls) = (
    s=[0.0];
    for λ in ls
        s = vcat(s, s .+ λ)
    end;
    sort(s)
)

@testset "method-recovery gate · Kondo model (U1U1)" begin
    # ---- (1) exact init spectrum: singlet −3J/4, triplet J/4 (×3), 0 (×4) ----
    @testset "init: singlet/triplet of S·s" begin
        J = 0.4
        ch = wilson_chain(WilsonLog(2.0), KondoModel(; J), 4)
        E = sort(reduce(vcat, values(impurity_init(KondoModel(; J), U1U1(), ch).E)))
        @test E ≈ sort([-3J / 4, 0, 0, 0, 0, J / 4, J / 4, J / 4])
    end

    # ---- (2) J=0 ⇒ free spin × free chain (exact) ----
    @testset "J=0: free spin × free conduction chain" begin
        nsites = 5
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )                    # keep-all ⇒ exact
        res = nrg_solve(KondoModel(; J=0.0), alg)
        got = sort(res.energies[end])
        sp = _chain_sp(res.chain, nsites)
        ss = _subset(vcat(sp, sp))                               # conduction ↑,↓
        ref = sort(vcat(ss, ss))                                 # × free impurity spin (×2)
        @test length(got) == length(ref) == 2048
        @test maximum(abs, (got .- got[1]) .- (ref .- ref[1])) < 1.0e-9
    end

    # ---- (3) J>0 ⇒ flow reaches the Kondo strong-coupling fixed point ----
    @testset "J>0: flow reaches a fixed point" begin
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(300), nsites=32
        )
        res = nrg_solve(KondoModel(; J=0.4), alg)
        @test all(isfinite, reduce(vcat, res.energies))
        lo(n) = sort(res.energies[n])[1:6]
        @test maximum(abs, lo(lastindex(res.energies)) .- lo(lastindex(res.energies) - 2)) <
            0.05
    end

    # ---- (4) thermodynamics is genuinely model-generic: Kondo screening ----
    # The impurity is a localized spin-½ (2 states): high-T S_imp → ln2 (NOT ln4 —
    # distinguishes it from Anderson's free orbital), then Kondo-screened to 0 at low T.
    @testset "Kondo screening (thermodynamics reused on KondoModel)" begin
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0),
            symmetry=U1U1(),
            truncation=EnergyCut(7.0),
            nsites=22,
        )
        th = thermodynamics(KondoModel(; J=0.3), alg; betabar=1.0)
        @test isapprox(th.S_imp[1], log(2); atol=0.05)   # high-T free spin: ln2 (2 states)
        @test all(≤(0.27), th.Tχ_imp)                     # free moment never exceeds ~1/4
        @test th.Tχ_imp[end] < 0.05                       # screened singlet (got ≈0.024)
        @test th.S_imp[end] < 0.20                        # entropy quenched (got ≈0.128)
    end
end
