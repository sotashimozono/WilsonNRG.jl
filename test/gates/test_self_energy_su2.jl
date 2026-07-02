# Faithfulness gate — the impurity self-energy on the U1SU2 (spin-SU(2)) engine, BOTH routes:
#
#  (Dyson) Σ = ω − εd − Δ − 1/G is a deterministic functional of G, and green_function(CFS)
#     reproduces the U1U1 G on the U1SU2 engine to machine precision (test_cfs_su2.jl), so the
#     U1SU2 Dyson self-energy reproduces the U1U1 one — a cross-symmetry identity.
#  (trick) Σ = U·F/G with F = ⟨⟨d_↑ n_↓; d†_↑⟩⟩ is the ACCURATE route: the compound operator
#     O_F = n_↓ d†_↑ is d†_↑ with the (0,0)→(1,½) block dropped (still a spin-½ tensor, propagated
#     like d†), and the per-spin CG weight cancels in F/G, so Σ is robust — it reproduces the
#     Fermi-liquid pin ReΣ(0)=U/2, ImΣ(0)=0 at the symmetric point, and Σ≡0 at U=0, INDEPENDENT
#     targets (Luttinger; Bulla–Hewson–Pruschke 1998) that the broadening-limited Dyson route
#     does not hit. (There is deliberately no standalone BHP spectral for U1SU2 — CFS is the
#     accurate U1SU2 A(ω); the windowed G here exists only inside the robust F/G ratio.)

using WilsonNRG, Test

@testset "faithfulness gate · self-energy on the U1SU2 engine (Dyson + trick)" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)

    # ---- (Dyson) keep-all: G matches to machine precision ⇒ Σ matches ----
    @testset "Dyson keep-all Σ_U1SU2 == Σ_U1U1 · nsites=$nsites" for nsites in (3, 4)
        alg1 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(10^9), nsites
        )
        r1 = self_energy(CFS(), m, alg1; via=Dyson())
        r2 = self_energy(CFS(), m, algs; via=Dyson(), ω=r1.ω)
        @test r2.ω == r1.ω
        @test maximum(abs, real.(r2.Σ .- r1.Σ)) < 1.0e-9
        @test maximum(abs, imag.(r2.Σ .- r1.Σ)) < 1.0e-9
    end

    # ---- (trick) Fermi-liquid pin ReΣ(0)=U/2, ImΣ(0)=0 at the symmetric point ----
    @testset "trick ReΣ(0)=U/2, ImΣ(0)=0 · U=$U" for U in (0.3, 0.5)
        mm = AndersonModel(; U, εd=(-U / 2), Γ=0.05, D=1.0)
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5),
            symmetry=U1SU2(),
            truncation=KeepN(300),
            nsites=26,
        )
        r = self_energy(BHP(), mm, algs)            # default via = trick; BHP builds the F-correlator
        k = argmin(abs.(r.ω))
        @test isapprox(real(r.Σ[k]), U / 2; atol=0.03)     # Luttinger pin
        @test isapprox(imag(r.Σ[k]), 0.0; atol=0.02)       # Fermi liquid
    end

    # ---- (trick) U=0 ⇒ Σ ≡ 0 exactly (Σ = U·F/G with U=0) ----
    @testset "trick U=0 ⇒ Σ=0" begin
        m0 = AndersonModel(; U=0.0, εd=0.0, Γ=0.05, D=1.0)
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5),
            symmetry=U1SU2(),
            truncation=KeepN(300),
            nsites=26,
        )
        r = self_energy(BHP(), m0, algs)
        @test maximum(abs, real.(r.Σ)) < 1.0e-10
    end

    # ---- honest refusals: the trick needs BHP (F-correlator); Dyson needs a supported symmetry ----
    @testset "honest refusals" begin
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(64), nsites=4
        )
        alg22 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=SU2SU2(), truncation=KeepN(64), nsites=4
        )
        @test_throws EngineUnimplemented self_energy(CFS(), m, algs; via=SelfEnergyTrick()) # trick needs BHP
        @test_throws EngineUnimplemented self_energy(CFS(), m, alg22; via=Dyson())          # SU2SU2 not wired
    end
end
