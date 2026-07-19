# ===========================================================================
#  Density-matrix NRG (DM-NRG) spectral function — Hofstetter, [doi_10.1103_PhysRevLett.85.1508](@cite).
#  Review: Bulla, Costi & Pruschke, [doi_10.1103_RevModPhys.80.395](@cite), §III.B.2, Eqs. 86–88.
#
#  Improves on the ground-state-projector complete-Fock-space method (cfs.jl) by using the
#  OFF-DIAGONAL reduced density matrix ρ_red(n) = Tr_{sites>n}[|G⟩⟨G|] (RMP Eq. 86), computed
#  by a backward sweep that RETAINS the coherences the CFS diagonal drops. The spectral weight
#  is the general-density-matrix Lehmann form: for the transition |i⟩(Q) ↔ |j⟩(Q+1) via d†_σ
#  (O = ⟨j|d†_σ|i⟩),
#      add    (ω = +ωₙ(E_j − E_i)):  w = O_{ji} · (O ρ^{(Q)})_{ji}
#      remove (ω = −ωₙ(E_j − E_i)):  w = O_{ji} · (ρ^{(Q+1)} O)_{ji}
#  which reduces to the CFS/FDM diagonal weight (ρ_i+ρ_j)|O|² when ρ is diagonal, and gives the
#  exact sum rule  ∫A_σ dω = Tr[ρ_Q d_σ d†_σ] + Tr[ρ_{Q+1} d†_σ d_σ] = ⟨{d_σ,d†_σ}⟩ = 1.
#  The retained coherences are what make DM-NRG correct where the ground-state projector fails
#  (e.g. the spin-resolved spectrum in a magnetic field; Hofstetter Fig. 11 / RMP Fig. 11).
# ===========================================================================

# off-diagonal reduced density matrices ρ_red(n) on each shell's kept states (Hermitian, PSD,
# trace 1), by the backward trace of |G⟩⟨G| (RMP Eq. 86). Full-matrix generalization of the CFS
# diagonal sweep; its diagonal reduces to `_cfs_reduced_dms` up to the retained coherences.
function _dmnrg_reduced_dms(shells::Vector{_CFSShell})
    N = length(shells)
    last = shells[N]
    gE = minimum(minimum(last.vals[qn][idx]) for (qn, idx) in last.plan)
    gstates = [
        (qn, p) for (qn, idx) in last.plan for
        (p, i) in enumerate(idx) if isapprox(last.vals[qn][i], gE; atol=1.0e-9)
    ]
    ρ = Vector{Dict{NTuple{2,Int},Matrix{Float64}}}(undef, N)
    ρ[N] = Dict(qn => zeros(length(idx), length(idx)) for (qn, idx) in last.plan)
    for (qn, p) in gstates                                   # |G⟩⟨G|, degenerate-averaged
        ρ[N][qn][p, p] = 1.0 / length(gstates)
    end
    for n in (N - 1):-1:1
        child = shells[n + 1]
        ρn = Dict(P => zeros(length(idx), length(idx)) for (P, idx) in shells[n].plan)
        for (nqn, idx) in child.plan
            Vk = child.vecs[nqn][:, idx]                     # kept eigenvectors of shell n+1
            ρc = ρ[n + 1][nqn]
            for (P, s, r) in child.seg[nqn]                  # parent P, new-site state s, rows r
                haskey(ρn, P) || continue
                M = ρn[P]
                @inbounds for k in eachindex(r), k2 in eachindex(r)
                    acc = 0.0
                    for a in axes(ρc, 1), a2 in axes(ρc, 2)  # trace the new site (fixed s)
                        ρc[a, a2] == 0.0 && continue
                        acc += Vk[r[k], a] * ρc[a, a2] * Vk[r[k2], a2]
                    end
                    M[k, k2] += acc
                end
            end
        end
        ρ[n] = ρn
    end
    return ρ
end

# not-both-kept DM-NRG poles (ω_phys, weight) for spin σ, with the general-ρ Lehmann weights.
function _dmnrg_poles(shells::Vector{_CFSShell}, ρ, σ::Int; tol::Real=1.0e-15)
    N = length(shells)
    poles = Tuple{Float64,Float64}[]
    for n in 1:N
        sh = shells[n]
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        for ((Q, D, σd), O) in sh.Ofull                     # O = ⟨tgt|d†_σ|src⟩ (full eigenbasis)
            σd == σ || continue
            src = (Q, D)
            tgt = (Q + 1, D + σ)
            (haskey(sh.plan, src) && haskey(sh.plan, tgt)) || continue
            ks = get(kept, src, Set{Int}())
            kt = get(kept, tgt, Set{Int}())
            Es = sh.vals[src]
            Et = sh.vals[tgt]
            # ADD (ω>0): kept src i → final tgt j (discarded); weight O_{ji}·(O ρ^{src})_{ji}
            if haskey(ρ[n], src)
                ρs = ρ[n][src]
                idxs = sh.plan[src]
                for (kp, ki) in enumerate(idxs), j in 1:length(Et)
                    (n == N || !(j in kt)) || continue       # j final unless last shell
                    Oji = O[j, ki]
                    Oji == 0.0 && continue
                    oρ = 0.0
                    @inbounds for (kp2, ki2) in enumerate(idxs)
                        oρ += O[j, ki2] * ρs[kp2, kp]         # (O ρ^{src})_{j,ki}
                    end
                    w = Oji * oρ
                    abs(w) < tol && continue
                    push!(poles, (sh.ω * (Et[j] - Es[ki]), w))
                end
            end
            # REMOVE (ω<0): kept tgt j → final src i (discarded); weight O_{ji}·(ρ^{tgt} O)_{ji}
            if haskey(ρ[n], tgt)
                ρt = ρ[n][tgt]
                idxt = sh.plan[tgt]
                for (kp, kj) in enumerate(idxt), i in 1:length(Es)
                    (n == N || !(i in ks)) || continue       # i final unless last shell
                    Oji = O[kj, i]
                    Oji == 0.0 && continue
                    ρo = 0.0
                    @inbounds for (kp2, kj2) in enumerate(idxt)
                        ρo += ρt[kp, kp2] * O[kj2, i]         # (ρ^{tgt} O)_{kj,i}
                    end
                    w = Oji * ρo
                    abs(w) < tol && continue
                    push!(poles, (sh.ω * (Et[kj] - Es[i]), w))
                end
            end
        end
    end
    return poles
end

"""
    green_function(::DMNRG, model::AndersonModel, alg; b=0.6, ω=nothing) -> (; ω, G)

Retarded impurity Green's function via the density-matrix NRG (Hofstetter, [doi_10.1103_PhysRevLett.85.1508](@cite);
Bulla–Costi–Pruschke [doi_10.1103_RevModPhys.80.395](@cite), Eqs. 86–88). Uses the **off-diagonal** reduced density
matrix of the ground state (the coherences the ground-state-projector [`CFS`](@ref) drops), giving
the exact sum rule `∫A_σ dω = 1` by the anticommutator. `U1U1`. `A(ω) = -Im G/π` is [`spectral`](@ref).
"""
function green_function(
    ::DMNRG, model::AndersonModel, alg::NRGAlgorithm; b::Real=0.6, ω=nothing
)
    # U1U1-only by design, same reason as FDM: the non-abelian finite-T reduced density matrix
    # needs the QSpace (2S+1) multiplet-weight bookkeeping (not yet machine-precise here). U1SU2
    # finite-T is deferred; use CFS (T=0, exact) + the self-energy trick for U1SU2 dynamics.
    alg.symmetry isa U1U1 || throw(
        EngineUnimplemented(
            "DMNRG green_function needs U1U1 (got $(typeof(alg.symmetry)))"
        ),
    )
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    shells = _cfs_collect(model, alg)
    ρ = _dmnrg_reduced_dms(shells)
    poles = _dmnrg_poles(shells, ρ, 1)                       # spin ↑ (per-spin A)
    return (; ω=ωs, G=_correlator(poles, ωs, b, 2))
end
