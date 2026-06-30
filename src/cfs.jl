# ===========================================================================
#  Complete-Fock-space (CFS) spectral function — the T=0 complete-basis method.
#
#  Peters, Pruschke & Anders, PRB 74, 245114 (2006); complete basis of Anders &
#  Schiller, PRL 95, 196801 (2005). The discarded states of every NRG shell, tensored
#  with the environment of the not-yet-added sites, form a COMPLETE basis of the full
#  chain Fock space:  Σ_n Σ_{s∈D_n} |s;e⟩⟨s;e| = 1.  Summing the Lehmann representation
#  over this basis makes the spectral sum rule ∫A_σ dω = ⟨G|{d_σ,d†_σ}|G⟩ = 1 hold
#  *exactly* (by completeness) — the qualitative gain over the `BHP` patching, whose
#  windowed tiling only approximates it.
#
#  At T=0 the density matrix is ρ = |G⟩⟨G| (global ground state). Its diagonal
#  reduced density matrices ρ_n[k] on each shell's kept space are built by a backward
#  sweep; the pole energies are within-shell differences ωₙ(E_s − E_k), so no
#  cross-shell absolute-energy bookkeeping is needed (that enters only at finite T —
#  the `FDM` generalization, Weichselbaum & von Delft, PRL 99, 076402 (2007)).
# ===========================================================================

# d†_σ rotated into a shell's FULL eigenbasis (all states, not only the kept ones —
# the discarded rows/cols are the spectral final states). `O` is d†_σ in the parent
# kept eigenbasis; mirrors `propagate_operator` but keeps every eigenvector column.
function _cfs_propagate_full(O::Dict{NTuple{3,Int},Matrix{Float64}}, diag::U1U1Diag)
    Vf = diag.vecs
    Onew = Dict{NTuple{3,Int},Matrix{Float64}}()
    for (qn, segs) in diag.seg
        haskey(Vf, qn) || continue
        Q, D = qn
        for σ in (1, -1)
            tgt = (Q + 1, D + σ)
            haskey(Vf, tgt) || continue
            M = zeros(Float64, size(Vf[tgt], 1), size(Vf[qn], 1))
            tgtseg = Dict((p, s) => r for (p, s, r) in diag.seg[tgt])
            for (P, s, r) in segs
                P′ = (P[1] + 1, P[2] + σ)
                (haskey(O, (P[1], P[2], σ)) && haskey(tgtseg, (P′, s))) || continue
                M[tgtseg[(P′, s)], r] = O[(P[1], P[2], σ)]
            end
            blk = transpose(Vf[tgt]) * M * Vf[qn]
            iszero(blk) || (Onew[(Q, D, σ)] = blk)
        end
    end
    return Onew
end

# per-shell data retained for the complete-basis sum + backward DM sweep
struct _CFSShell
    vals::Dict{NTuple{2,Int},Vector{Float64}}   # all eigenvalues per (Q,D) block
    vecs::Dict{NTuple{2,Int},Matrix{Float64}}   # all eigenvectors (for the backward sweep)
    seg::Dict{NTuple{2,Int},Vector{_Seg}}       # product-basis segmentation
    plan::Dict{NTuple{2,Int},Vector{Int}}       # kept indices per block
    Ofull::Dict{NTuple{3,Int},Matrix{Float64}}  # d†_σ in the full eigenbasis
    ω::Float64                                   # shell scale ωₙ
end

# run the flow, retaining each shell's full decomposition + the full-basis d†
function _cfs_collect(model::AbstractImpurityModel, alg::NRGAlgorithm)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sym = alg.symmetry
    sqrtΛ = sqrt(alg.discretization.Λ)
    st = impurity_init(model, sym, chain)
    O = deepcopy(st.F)                                   # impurity d†_σ
    shells = _CFSShell[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, sym; coupling, rescale, onsite=chain.onsite[n + 1]), sym
        )
        plan = truncation_plan(diag.vals, alg.truncation, sym)
        push!(
            shells,
            _CFSShell(
                deepcopy(diag.vals),
                deepcopy(diag.vecs),
                deepcopy(diag.seg),
                deepcopy(plan),
                _cfs_propagate_full(O, diag),
                shell_scale(alg.discretization, n),
            ),
        )
        O = propagate_operator(O, diag, plan, sym)       # kept d†_σ for the next shell
        st = update_operators(diag, plan, sym)
    end
    return shells
end

# diagonal reduced density matrices ρ_n[block][kept-position] of the global ground state,
# by the backward DM-NRG sweep. ρ_N = projector on the global ground state; then
# ρ_n[P][k] = Σ_{nqn,a,s} |Vk_{n+1}[nqn][row(P,k,s), a]|² ρ_{n+1}[nqn][a] (trace the new site).
function _cfs_reduced_dms(shells::Vector{_CFSShell})
    N = length(shells)
    ρ = Vector{Dict{NTuple{2,Int},Vector{Float64}}}(undef, N)
    last = shells[N]
    gE = minimum(minimum(last.vals[qn][idx]) for (qn, idx) in last.plan)
    ρ[N] = Dict(qn => zeros(length(idx)) for (qn, idx) in last.plan)
    # Split unit weight over the (possibly degenerate) ground multiplet — the T→0⁺ average.
    # A strict-min single pick would be Dict-order-dependent when the ground state is
    # degenerate (e.g. the odd-parity Kondo doublet at U>0); per-spin sum rules survive
    # either way, but the split is deterministic and gives the correct degenerate average.
    gstates = [
        (qn, p) for (qn, idx) in last.plan for
        (p, i) in enumerate(idx) if isapprox(last.vals[qn][i], gE; atol=1e-9)
    ]
    for (qn, p) in gstates
        ρ[N][qn][p] = 1.0 / length(gstates)
    end
    for n in (N - 1):-1:1
        child = shells[n + 1]
        ρn = Dict(qn => zeros(length(idx)) for (qn, idx) in shells[n].plan)
        for (nqn, idx) in child.plan
            Vk = child.vecs[nqn][:, idx]                 # kept eigenvectors of shell n+1
            w = ρ[n + 1][nqn]
            for (P, s, r) in child.seg[nqn]              # parent block P, new-site state s, rows r
                haskey(ρn, P) || continue
                ρP = ρn[P]
                for k in eachindex(r)
                    row = r[k]
                    acc = 0.0
                    @inbounds for a in eachindex(w)
                        w[a] == 0.0 && continue
                        acc += abs2(Vk[row, a]) * w[a]
                    end
                    ρP[k] += acc
                end
            end
        end
        ρ[n] = ρn
    end
    return ρ
end

# complete-basis poles (ω_phys, weight, 0.0) for spin σ (formatted for `_correlator`).
function _cfs_poles(shells::Vector{_CFSShell}, ρ, σ::Int)
    N = length(shells)
    poles = Tuple{Float64,Float64,Float64}[]
    for n in 1:N
        sh = shells[n]
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        final(qn, i) = n == N || !(i in get(kept, qn, Set{Int}()))   # last shell: all states final
        # ADD (ω>0): |G⟩(kept k, charge Q) → discarded s (charge Q+1)
        for (qn, idx) in sh.plan
            Q, D = qn
            tgt = (Q + 1, D + σ)
            (haskey(sh.Ofull, (Q, D, σ)) && haskey(sh.vals, tgt)) || continue
            Ob = sh.Ofull[(Q, D, σ)]                                 # ⟨tgt|d†_σ|qn⟩
            for (kp, ki) in enumerate(idx)
                w0 = ρ[n][qn][kp]
                w0 < 1.0e-15 && continue
                Ek = sh.vals[qn][ki]
                for s in 1:length(sh.vals[tgt])
                    final(tgt, s) || continue
                    g = Ob[s, ki]
                    g == 0.0 && continue
                    push!(poles, (sh.ω * (sh.vals[tgt][s] - Ek), w0 * g^2, 0.0))
                end
            end
        end
        # REMOVE (ω<0): |G⟩(kept k, charge Q+1) → discarded s (charge Q); ω = E_G − E_s < 0
        for (qn, idx) in sh.plan
            Q, D = qn
            tgt = (Q + 1, D + σ)
            (haskey(sh.Ofull, (Q, D, σ)) && haskey(sh.plan, tgt)) || continue
            Ob = sh.Ofull[(Q, D, σ)]
            for (kp, ki) in enumerate(sh.plan[tgt])
                w0 = ρ[n][tgt][kp]
                w0 < 1.0e-15 && continue
                Ek = sh.vals[tgt][ki]
                for s in 1:length(sh.vals[qn])
                    final(qn, s) || continue
                    g = Ob[ki, s]
                    g == 0.0 && continue
                    push!(poles, (sh.ω * (Ek - sh.vals[qn][s]), w0 * g^2, 0.0))
                end
            end
        end
    end
    return poles
end

"""
    green_function(::CFS, model::AndersonModel, alg; b=0.6, ω=nothing) -> (; ω, G)

Retarded impurity Green's function via the complete-Fock-space (CFS) method at T=0:
the Lehmann sum over the complete basis of discarded states (Peters–Pruschke–Anders
2006; Anders–Schiller 2005). The spectral sum rule `∫A_σ dω = 1` holds *exactly* by
completeness — the gain over [`BHP`](@ref) patching (which only approximates it).
Needs `U1U1`. `A(ω) = -Im G/π` is [`spectral`](@ref).
"""
function green_function(
    ::CFS, model::AndersonModel, alg::NRGAlgorithm; b::Real=0.6, ω=nothing
)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("CFS green_function needs U1U1 (got $(typeof(alg.symmetry)))"))
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    shells = _cfs_collect(model, alg)
    ρ = _cfs_reduced_dms(shells)
    poles = _cfs_poles(shells, ρ, 1)                     # spin ↑ (per-spin A; symmetric point)
    return (; ω=ωs, G=_correlator(poles, ωs, b, 2))
end
