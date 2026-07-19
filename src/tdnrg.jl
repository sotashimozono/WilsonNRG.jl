# ===========================================================================
#  Time-dependent NRG (TDNRG) — real-time impurity dynamics after a quench.
#
#  Anders & Schiller, [doi_10.1103_PhysRevLett.95.196801](@cite): the impurity parameters are switched suddenly at
#  t=0 (H_i → H_f, sharing the Wilson chain), and the impurity observable evolves as
#       ⟨O(t)⟩ = ⟨G_i| e^{iH_f t} O e^{-iH_f t} |G_i⟩.
#  Two routes, selected by the truncation:
#
#  • KEEP-ALL (`KeepN(typemax(Int))`, short chains) — exact. |G_i⟩ is expressed in the full H_f
#    eigenbasis by the shell overlap recursion S_n = V_f^⊤ (S_{n-1} ⊗ I_site) V_i, and
#    ⟨O(t)⟩ = Σ_{s,s'} ⟨G_i|s⟩⟨s'|G_i⟩ e^{i(E_s−E_{s'})t} ⟨s|O|s'⟩ over the last shell.
#
#  • TRUNCATED COMPLETE BASIS (any real `KeepN`, scalable to long chains) — Anders–Schiller
#    Eq. (3): ⟨O⟩(t) = Σ_{m=0}^N Σ_{r,s}^{trun} e^{i(E_r^m−E_s^m)t} O^m_{r,s} ρ^red_{s,r}(m), where
#    Σ^trun runs over pairs with at least one of r,s DISCARDED at shell m (all states discarded at
#    the last shell), O^m is the impurity observable in the shell-m H_f eigenbasis, and the
#    reduced density matrix (Eq. 4) ρ^red_{s,r}(m) = Σ_e ⟨s,e;m|ρ_eq|r,e;m⟩.
#
#    Construction (reuses the DM-NRG off-diagonal reduced-DM machinery, #20): the initial state
#    |G_i⟩ lives entirely in the ITERATED KEPT subspace of the H_i run (the NRG discards states,
#    so |G_i⟩ has ZERO weight on H_i's own discarded states), hence its reduced density matrices
#    ρ^i(m) = `_dmnrg_reduced_dms` of the H_i run are complete on the kept space. The weight |G_i⟩
#    carries on H_f's DISCARDED states — which makes Σ^trun non-trivial — comes purely from the
#    basis ROTATION:  ρ^red_f(m) = S_ki(m) · ρ^i(m) · S_ki(m)^⊤,  where S_ki(m) is the overlap of
#    every H_f state at shell m with the kept H_i states (all_f × kept_i). Exact by the identity
#    ρ^red_f(m)_{s,r} = Σ_{a,b} S(s,a) ρ^i(m)_{a,b} S(r,b) once |G_i⟩ ⟂ discarded_i.
#
#  Energies: the shell-m physical eigenenergy is E^m = ωₘ · (rescaled), ωₘ = Λ^{-(n)/2} for the
#  physical site n = bath_sites_in_init + (m−1) (the same Λ^{n/2} rescaling the free-fermion gate
#  verifies; NOT `shell_scale`, which carries an extra (1+Λ⁻¹)/2 broadening-window factor). The
#  Eq. (3) sum is diagonal in the shell m, so only within-shell differences enter — the
#  per-shell ground-subtraction cancels and no cross-shell absolute-energy bookkeeping is needed.
#
#  Faithfulness (test/gates/test_tdnrg.jl): U=0 ⇒ ⟨n_d(t)⟩ = the single-particle quench (exact,
#  independent) — exact at keep-all, and convergent under truncation as KeepN↑; t=0 ⇒ ⟨n_d(0)⟩ =
#  occupation(initial) (completeness); truncated == keep-all in the exact limit. z-averaging and
#  the Eq. (5) damping (relaxation of a genuinely infinite bath) are further refinements.
# ===========================================================================

using LinearAlgebra: I

_is_keepall(t::KeepN) = t.N == typemax(Int)
_is_keepall(::AbstractTruncation) = false

# one overlap-recursion step:  S_n[qn] = V_f[qn]^⊤ (S_{n-1} ⊗ I_site) V_i[qn], block-diagonal in qn.
# At keep-all the two runs share the product-basis segmentation (same kept dims), so the embedded
# S_{n-1} sits block-diagonally on the parent-block ranges (identity on the new site).
function _overlap_step(S, df::U1U1Diag, di::U1U1Diag)
    Snew = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (qn, segs) in df.seg
        Vf = df.vecs[qn]
        Vi = di.vecs[qn]
        M = zeros(Float64, size(Vf, 1), size(Vi, 1))
        for (p, s, r) in segs
            haskey(S, p) && (M[r, r] = S[p])          # S_{n-1}[parent p] ⊗ I on site s
        end
        Snew[qn] = transpose(Vf) * M * Vi
    end
    return Snew
end

# --- exact keep-all path: |G_i⟩ in the full last-shell H_f basis, single-shell phase sum. ---
function _quench_keepall(
    initial::AndersonModel, final::AndersonModel, alg::NRGAlgorithm, times
)
    chain = wilson_chain(alg.discretization, final, alg.nsites)
    Λ = alg.discretization.Λ
    sqrtΛ = sqrt(Λ)
    stf = impurity_init(final, U1U1(), chain)
    sti = impurity_init(initial, U1U1(), chain)
    S = Dict(qn => Matrix{Float64}(I, length(ev), length(ev)) for (qn, ev) in stf.E)
    nd = Dict(
        (1, 1) => fill(1.0, 1, 1), (1, -1) => fill(1.0, 1, 1), (2, 0) => fill(2.0, 1, 1)
    )
    keepall = KeepN(typemax(Int))
    for n in bath_sites_in_init(final):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(final) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        df = diagonalize_blocks(
            add_site(stf, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        di = diagonalize_blocks(
            add_site(sti, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        S = _overlap_step(S, df, di)
        pf = truncation_plan(df.vals, keepall, U1U1())
        pit = truncation_plan(di.vals, keepall, U1U1())
        nd = propagate_observable(nd, df, pf, U1U1())
        stf = update_operators(df, pf, U1U1())
        sti = update_operators(di, pit, U1U1())
    end
    # global H_i ground state (energies are ground-subtracted each shell ⇒ min ≈ 0)
    gE = minimum(minimum(v) for v in values(sti.E))
    blkg = first(qn for (qn, ev) in sti.E if any(e -> isapprox(e, gE; atol=1.0e-9), ev))
    gi = findfirst(e -> isapprox(e, gE; atol=1.0e-9), sti.E[blkg])
    a = S[blkg][:, gi]                                    # ⟨s_f | G_i⟩ for H_f states s in blkg
    E = stf.E[blkg]                                       # H_f rescaled energies (same block)
    O = get(nd, blkg, zeros(length(a), length(a)))        # n_d in the H_f eigenbasis
    scale = Λ^(-(alg.nsites - 1) / 2)                     # rescaled ΔE → physical
    ts = collect(float.(times))
    ndt = [
        real(
            sum(
                a[s] * a[sp] * cis((E[s] - E[sp]) * scale * t) * O[s, sp] for
                s in eachindex(a), sp in eachindex(a)
            ),
        ) for t in ts
    ]
    return (; t=ts, nd=ndt)
end

# --- truncated complete-basis path (Anders–Schiller Eq. 3/4) ---

# n_d in the FULL shell-m H_f eigenbasis (kept AND discarded rows/cols — the discarded states are
# the complete-basis final states of Eq. 3). n_d acts on the impurity only (a spectator on every
# bath site), so it embeds diagonally: full = V^⊤ (n_d^kept-parent ⊗ I_site) V. Forward sweep.
function _tdnrg_nd_full(shells::Vector{_CFSShell})
    N = length(shells)
    ndk = Dict(
        (0, 0) => zeros(1, 1),
        (1, 1) => fill(1.0, 1, 1),
        (1, -1) => fill(1.0, 1, 1),
        (2, 0) => fill(2.0, 1, 1),
    )                                                    # impurity n_d = diag(0, 1, 1, 2)
    O = Vector{Dict{NTuple{2,Int},Matrix{Float64}}}(undef, N)
    for m in 1:N
        sh = shells[m]
        Om = Dict{NTuple{2,Int},Matrix{Float64}}()
        ndknew = Dict{NTuple{2,Int},Matrix{Float64}}()
        for (qn, segs) in sh.seg
            V = sh.vecs[qn]
            M = zeros(size(V, 1), size(V, 1))
            for (P, s, r) in segs
                haskey(ndk, P) && (M[r, r] = ndk[P])     # n_d^parent ⊗ I on the new site
            end
            full = transpose(V) * M * V
            Om[qn] = full
            kf = get(sh.plan, qn, Int[])
            ndknew[qn] = full[kf, kf]                     # kept n_d for the next shell's embed
        end
        O[m] = Om
        ndk = ndknew
    end
    return O
end

# per-shell overlap S_ki(m) = ⟨all H_f state ; m | kept H_i state ; m⟩ (all_f × kept_i), block
# diagonal in the conserved (Q,D). Both runs are built from their OWN kept parents, so the
# recursion carries the kept_f × kept_i block forward (the rectangular S under truncation).
function _tdnrg_overlaps(shells_f::Vector{_CFSShell}, shells_i::Vector{_CFSShell}, Simp)
    N = length(shells_f)
    Sp = Simp                                            # impurity-level parent overlap (= I)
    Ski = Vector{Dict{NTuple{2,Int},Matrix{Float64}}}(undef, N)
    for m in 1:N
        shf = shells_f[m]
        shi = shells_i[m]
        S = Dict{NTuple{2,Int},Matrix{Float64}}()
        Snext = Dict{NTuple{2,Int},Matrix{Float64}}()
        for (qn, segf) in shf.seg
            haskey(shi.vecs, qn) || continue
            kept_i = get(shi.plan, qn, Int[])            # a block may keep 0 states under trunc
            isempty(kept_i) && continue
            Vf = shf.vecs[qn]
            Vi = shi.vecs[qn]
            M = zeros(size(Vf, 1), size(Vi, 1))          # product basis: prod_f × prod_i
            segi = Dict((P, s) => r for (P, s, r) in shi.seg[qn])
            for (P, s, rf) in segf
                (haskey(Sp, P) && haskey(segi, (P, s))) || continue
                M[rf, segi[(P, s)]] = Sp[P]              # S_{m-1}[parent P] on matching (P, site s)
            end
            Sqn = transpose(Vf) * M * Vi[:, kept_i]      # all_f × kept_i
            S[qn] = Sqn
            Snext[qn] = Sqn[get(shf.plan, qn, Int[]), :] # kept_f × kept_i, carried forward
        end
        Ski[m] = S
        Sp = Snext
    end
    return Ski
end

function _quench_complete_basis(
    initial::AndersonModel, final::AndersonModel, alg::NRGAlgorithm, times
)
    Λ = alg.discretization.Λ
    shells_i = _cfs_collect(initial, alg)
    shells_f = _cfs_collect(final, alg)
    ρi = _dmnrg_reduced_dms(shells_i)                    # kept_i × kept_i, H_i basis (exact on kept)
    chain = wilson_chain(alg.discretization, initial, alg.nsites)
    sti0 = impurity_init(initial, U1U1(), chain)
    Simp = Dict(qn => Matrix{Float64}(I, length(ev), length(ev)) for (qn, ev) in sti0.E)
    Ski = _tdnrg_overlaps(shells_f, shells_i, Simp)
    O = _tdnrg_nd_full(shells_f)
    N = length(shells_f)
    n0 = bath_sites_in_init(final)
    # ρ_red^f(m) = S_ki(m) ρ^i(m) S_ki(m)^⊤  (all_f × all_f)
    ρf = Vector{Dict{NTuple{2,Int},Matrix{Float64}}}(undef, N)
    for m in 1:N
        d = Dict{NTuple{2,Int},Matrix{Float64}}()
        for (qn, S) in Ski[m]
            haskey(ρi[m], qn) || continue
            d[qn] = S * ρi[m][qn] * transpose(S)
        end
        ρf[m] = d
    end
    # Eq. (3): Σ_m Σ_{r,s not both kept} cos(ωₘ(E_s−E_r)t) O^m_{s,r} ρ^red_f(m)_{r,s}
    # (real: n_d and ρ are real-symmetric, so the (s,r)+(r,s) phases combine to a cosine).
    ts = collect(float.(times))
    ndt = zeros(length(ts))
    for m in 1:N
        sh = shells_f[m]
        ω = Λ^(-(n0 + m - 1) / 2)                        # physical scale of shell m
        for (qn, Om) in O[m]
            haskey(ρf[m], qn) || continue
            ρ = ρf[m][qn]
            E = sh.vals[qn]
            kept = Set(get(sh.plan, qn, Int[]))
            nst = length(E)
            for s in 1:nst, r in 1:nst
                (m == N || !(s in kept) || !(r in kept)) || continue   # trun: ≥1 discarded
                Osr = Om[s, r]
                Osr == 0.0 && continue
                ρrs = ρ[r, s]
                ρrs == 0.0 && continue
                dω = ω * (E[s] - E[r])
                w = Osr * ρrs
                @inbounds for it in eachindex(ts)
                    ndt[it] += cos(dω * ts[it]) * w
                end
            end
        end
    end
    return (; t=ts, nd=ndt)
end

"""
    quench_dynamics(initial::AndersonModel, final::AndersonModel, alg; times) -> (; t, nd)

Real-time impurity occupation `⟨n_d(t)⟩` after a sudden quench of the impurity parameters from
`initial` to `final` at `t=0` — time-dependent NRG (Anders & Schiller, [doi_10.1103_PhysRevLett.95.196801](@cite)).
The system starts in the ground state of `initial` and evolves under `final`; `times` are the
(physical) times at which `⟨n_d⟩` is returned.

`initial` and `final` must share the bath (`Γ`, `D`) so the Wilson chain is common; they may
differ in `εd` and/or `U`. `U1U1` only.

Two routes, selected by `alg.truncation`:

- `KeepN(typemax(Int))` — the **exact keep-all** evaluation on a short chain (`alg.nsites` ≲ 7):
  |G_i⟩ in the full last-shell H_f basis, single-shell phase sum. Assumes a **non-degenerate**
  `initial` ground state (a single `findfirst` pick).
- any real `KeepN` — the **truncated complete-basis** method (Anders–Schiller Eq. 3, scalable to
  long chains): the discarded-state complete-Fock-space sum with the off-diagonal reduced density
  matrix (reuses the DM-NRG machinery, [`green_function`](@ref)`(::DMNRG, …)`). Reduces to the
  keep-all answer as `KeepN`↑; the residual at fixed `KeepN` is the NRG truncation error (which
  the paper further reduces by z-averaging and the Eq. 5 damping). This path is
  **degenerate-ground-state-safe** — it inherits the DM-NRG ensemble average (the `T→0⁺` mixed
  state over a degenerate `initial` multiplet), so it may legitimately differ from the keep-all
  path when the `initial` ground state is degenerate.

Checks: `⟨n_d(0)⟩ = ⟨n_d⟩` of `initial` ([`occupation`](@ref), the overlap is complete); at U=0
`⟨n_d(t)⟩` is the exact single-particle quench dynamics; the long-time signal relaxes toward the
`final` equilibrium up to finite-chain recurrences.
"""
function quench_dynamics(
    initial::AndersonModel, final::AndersonModel, alg::NRGAlgorithm; times
)
    alg.symmetry isa U1U1 || throw(
        EngineUnimplemented("quench_dynamics needs U1U1 (got $(typeof(alg.symmetry)))")
    )
    (initial.Γ == final.Γ && initial.D == final.D) || throw(
        ArgumentError(
            "quench_dynamics: initial and final must share the bath (Γ, D) — the Wilson chain is common",
        ),
    )
    # both runs must index the chain identically: shell m ↔ the same physical site n in each run,
    # so the shared ωₘ = Λ^{-(n0+m-1)/2} energy scale is well-defined (holds for any AndersonModel
    # pair today; guards a future signature that mixed impurity types with different bath-in-init).
    bath_sites_in_init(initial) == bath_sites_in_init(final) || throw(
        ArgumentError(
            "quench_dynamics: initial and final must have the same bath_sites_in_init (shared chain indexing)",
        ),
    )
    return if _is_keepall(alg.truncation)
        _quench_keepall(initial, final, alg, times)
    else
        _quench_complete_basis(initial, final, alg, times)
    end
end
