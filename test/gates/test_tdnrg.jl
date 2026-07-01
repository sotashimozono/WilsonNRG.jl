# Faithfulness gate — time-dependent NRG (Anders & Schiller, PRL 95, 196801 (2005)): real-time
# impurity dynamics ⟨n_d(t)⟩ after a sudden quench of the impurity parameters (H_i → H_f, shared
# chain), via the complete-basis time evolution with the H_i→H_f overlap recursion.
#  (1) U=0 ⇒ ⟨n_d(t)⟩ is EXACTLY the single-particle quench dynamics of the Wilson chain
#      (ρ(t)=e^{-iHf t}ρ0 e^{iHf t}, independently diagonalized) — the dynamical analogue of the
#      free-fermion / occupation gates; tier-1, at every time.
#  (2) t=0 ⇒ ⟨n_d(0)⟩ = ⟨n_d⟩ of the initial model (occupation) — the overlap is complete.
#  (3) no quench (initial=final) ⇒ ⟨n_d(t)⟩ is constant (= equilibrium occupation): no dynamics.
#  (4) shared-bath contract + honest EngineUnimplemented stub for U1SU2.

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigen, Diagonal

# independent single-particle quench reference (physical chain; exact U=0 many-body ⟨n_d(t)⟩)
function _sp_quench(εi, εf, chain, nsites, V0, Λ)
    ts = [chain.hopping[n] * Λ^(-n / 2) for n in 1:(nsites - 1)]
    L = nsites + 1
    function spmat(εd)
        H = zeros(L, L)
        H[1, 1] = εd
        H[1, 2] = H[2, 1] = V0
        for n in 1:(nsites - 1)
            H[n + 1, n + 2] = H[n + 2, n + 1] = ts[n]
        end
        return H
    end
    Fi = eigen(Symmetric(spmat(εi)))
    Ff = eigen(Symmetric(spmat(εf)))
    occ = Fi.values .< 0
    ρ0 = Fi.vectors[:, occ] * Fi.vectors[:, occ]'
    Uf(t) = Ff.vectors * Diagonal(cis.(-Ff.values .* t)) * Ff.vectors'
    return t -> 2 * real((Uf(t) * ρ0 * Uf(t)')[1, 1])
end

@testset "method-recovery gate · time-dependent NRG (Anders–Schiller quench dynamics)" begin
    Γ = 0.05
    D = 1.0
    Λ = 2.5
    nsites = 5
    alg = NRGAlgorithm(;                                          # keep-all: the exact short-chain path
        discretization=WilsonLog(Λ),
        symmetry=U1U1(),
        truncation=KeepN(typemax(Int)),
        nsites,
    )
    chain = wilson_chain(WilsonLog(Λ), AndersonModel(; U=0.0, εd=0.0, Γ, D), nsites)
    V0 = bath_coupling(AndersonModel(; U=0.0, εd=0.0, Γ, D))
    tgrid = [0.0, 1.0, 3.0, 7.0, 15.0, 30.0]

    # ---- (1) U=0 ⇒ exact single-particle quench dynamics, at every time ----
    @testset "U=0 single-particle reproduction" begin
        for (εi, εf) in ((-0.3, 0.2), (0.4, -0.1), (0.25, -0.25))
            got = quench_dynamics(
                AndersonModel(; U=0.0, εd=εi, Γ, D),
                AndersonModel(; U=0.0, εd=εf, Γ, D),
                alg;
                times=tgrid,
            )
            ref = _sp_quench(εi, εf, chain, nsites, V0, Λ)
            @test maximum(abs, got.nd .- ref.(tgrid)) < 1.0e-9
        end
    end

    # ---- (2) t=0 ⇒ ⟨n_d(0)⟩ = occupation(initial) (complete overlap), U=0 and U>0 ----
    @testset "t=0 equals the initial occupation" begin
        for (U, εi, εf) in ((0.0, -0.3, 0.2), (0.6, -0.3, 0.0), (0.4, -0.2, -0.2))
            init = AndersonModel(; U, εd=εi, Γ, D)
            q = quench_dynamics(init, AndersonModel(; U, εd=εf, Γ, D), alg; times=[0.0])
            @test q.nd[1] ≈ occupation(init, alg).total atol = 1.0e-9
        end
    end

    # ---- (3) no quench (initial = final) ⇒ static equilibrium, no time dependence ----
    @testset "no quench ⇒ constant equilibrium" begin
        m = AndersonModel(; U=0.5, εd=-0.25, Γ, D)
        q = quench_dynamics(m, m, alg; times=tgrid)
        @test maximum(q.nd) - minimum(q.nd) < 1.0e-9              # ⟨n_d(t)⟩ time-independent
        @test q.nd[1] ≈ occupation(m, alg).total atol = 1.0e-9   # = equilibrium occupation
    end

    # ---- (4) interacting quench: bounded, starts at the initial occupation ----
    @testset "interacting quench is physical" begin
        init = AndersonModel(; U=0.5, εd=-0.4, Γ, D)
        q = quench_dynamics(init, AndersonModel(; U=0.5, εd=0.1, Γ, D), alg; times=tgrid)
        @test all(-1.0e-9 .≤ q.nd .≤ 2.0 + 1.0e-9)               # 0 ≤ ⟨n_d⟩ ≤ 2
        @test q.nd[1] ≈ occupation(init, alg).total atol = 1.0e-9
    end

    # ---- (5) truncated complete-basis (Anders–Schiller Eq. 3): the scalable long-chain method ----
    @testset "truncated complete-basis reduces to keep-all and converges" begin
        _init(εd) = AndersonModel(; U=0.0, εd, Γ, D)
        # (a) exact limit: the complete-basis path (a real KeepN that still keeps everything on a
        #     short chain) reproduces the keep-all path bit-for-bit — the machinery is correct.
        ns = 4
        algK = NRGAlgorithm(;                                   # keep-all path (KeepN sentinel)
            discretization=WilsonLog(Λ),
            symmetry=U1U1(),
            truncation=KeepN(typemax(Int)),
            nsites=ns,
        )
        algC = NRGAlgorithm(;                                   # complete-basis path, keeps all here
            discretization=WilsonLog(Λ),
            symmetry=U1U1(),
            truncation=KeepN(10^6),
            nsites=ns,
        )
        for (εi, εf) in ((-0.3, 0.2), (0.25, -0.25))
            qk = quench_dynamics(_init(εi), _init(εf), algK; times=tgrid)
            qc = quench_dynamics(_init(εi), _init(εf), algC; times=tgrid)
            @test maximum(abs, qk.nd .- qc.nd) < 1.0e-9        # complete-basis ≡ keep-all (exact limit)
        end
        # (b) truncated LONG chain converges to the exact single-particle quench as KeepN ↑.
        nL = 14
        chainL = wilson_chain(WilsonLog(Λ), _init(0.0), nL)
        refL = _sp_quench(-0.3, 0.2, chainL, nL, V0, Λ)
        tg = [0.0, 2.0, 5.0, 10.0]
        alg_lo = NRGAlgorithm(;
            discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(50), nsites=nL
        )
        alg_hi = NRGAlgorithm(;
            discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(150), nsites=nL
        )
        q_lo = quench_dynamics(_init(-0.3), _init(0.2), alg_lo; times=tg)
        q_hi = quench_dynamics(_init(-0.3), _init(0.2), alg_hi; times=tg)
        d_lo = maximum(abs, q_lo.nd .- refL.(tg))
        d_hi = maximum(abs, q_hi.nd .- refL.(tg))
        @test d_hi < d_lo                                      # truncation error shrinks with KeepN
        @test d_hi < 0.12                                      # tracks the exact answer (single z;
        #                                        z-averaging + the Eq. 5 damping tighten it further)
        # t=0 completeness on the truncated long chain: ⟨n_d(0)⟩ = occupation(initial)
        @test q_hi.nd[1] ≈ occupation(_init(-0.3), alg_hi).total atol = 0.03
    end

    # ---- contracts: shared bath required; honest stub for unwired symmetry ----
    @testset "contracts" begin
        @test_throws ArgumentError quench_dynamics(
            AndersonModel(; U=0.0, εd=-0.3, Γ=0.05, D),
            AndersonModel(; U=0.0, εd=0.2, Γ=0.1, D),
            alg;
            times=[0.0],
        )
        alg_su2 = NRGAlgorithm(; discretization=WilsonLog(Λ), symmetry=U1SU2(), nsites)
        @test_throws EngineUnimplemented quench_dynamics(
            AndersonModel(; U=0.0, εd=-0.3, Γ, D),
            AndersonModel(; U=0.0, εd=0.2, Γ, D),
            alg_su2;
            times=[0.0],
        )
    end
end
