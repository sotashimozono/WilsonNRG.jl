# ===========================================================================
#  Complete-Fock-space (CFS) spectral function for the U(1)charge × SU(2)spin (`U1SU2`)
#  engine. The spectral function A(ω) is a physical observable and therefore INDEPENDENT of
#  the symmetry the engine exploits: this reproduces the `U1U1` CFS result (cfs.jl) to machine
#  precision at keep-all, and converges to it under truncation (the residual is the benign
#  keep-N-multiplets vs keep-N-states difference, which vanishes as N→∞) — the faithfulness gate
#  in test/gates/test_cfs_su2.jl.
#
#  Method (same Anders–Schiller / Peters–Pruschke–Anders completeness as cfs.jl): the impurity d†
#  is propagated as a spin-½ TENSOR operator (`propagate_impurity_op`, engine_u1su2.jl) whose
#  reduced matrix elements carry the spectral weight; the diagonal multiplet reduced density
#  matrices of the global ground state are built by a probability-conserving backward sweep; the
#  poles are within-shell energy differences with the SU(2) Clebsch–Gordan multiplicity weights.
# ===========================================================================

# per-shell data retained for the complete-basis sum + backward DM sweep (the U1SU2 analogue of
# `_CFSShell`; multiplets are (Q,S)-keyed, S a Rational).
struct _CFSShellSU2
    vals::Dict{Tuple{Int,Rational{Int}},Vector{Float64}}
    vecs::Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}
    seg::Dict{Tuple{Int,Rational{Int}},Vector{_SU2Seg}}
    plan::Dict{Tuple{Int,Rational{Int}},Vector{Int}}
    Ofull::Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}
    ω::Float64
end

# restrict a full-eigenbasis impurity-op to the kept (plan) rows/cols → the parent op for the next
# shell (the kept d† is exactly the full d† on the retained multiplets).
function _restrict_impurity_op(Ofull, plan)
    O = Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}()
    for ((Q, S, Sp), blk) in Ofull
        (haskey(plan, (Q, S)) && haskey(plan, (Q + 1, Sp))) || continue
        sub = blk[plan[(Q + 1, Sp)], plan[(Q, S)]]
        iszero(sub) || (O[(Q, S, Sp)] = sub)
    end
    return O
end

# run the flow, retaining each shell's full multiplet decomposition + the full-basis impurity d†
function _cfs_collect_su2(model::AbstractImpurityModel, alg::NRGAlgorithm)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sqrtΛ = sqrt(alg.discretization.Λ)
    st = impurity_init(model, U1SU2(), chain)
    O = deepcopy(st.F)                                   # impurity d† reduced ME (parent = impurity)
    shells = _CFSShellSU2[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, U1SU2(); coupling, rescale, onsite=chain.onsite[n + 1]), U1SU2()
        )
        plan = truncation_plan(diag.vals, alg.truncation, U1SU2())
        Ofull = propagate_impurity_op(O, diag)
        push!(
            shells,
            _CFSShellSU2(
                deepcopy(diag.vals),
                deepcopy(diag.vecs),
                deepcopy(diag.seg),
                deepcopy(plan),
                Ofull,
                shell_scale(alg.discretization, n),
            ),
        )
        O = _restrict_impurity_op(Ofull, plan)           # kept d† for the next shell
        st = update_operators(diag, plan, U1SU2())
    end
    return shells
end

# diagonal multiplet reduced density matrices ρ_n[(Q,S)][k] of the global ground state, by the
# backward sweep. PROBABILITY-CONSERVING (each shell Σρ = 1): the eigenvector overlaps already
# carry the multiplet structure, so tracing the new site needs NO extra (2S+1) weight (a
# (2S+1)/(2Sk+1) factor over-sums ∫A). Split unit weight over the (possibly degenerate) ground
# multiplet — the T→0⁺ average, deterministic under Dict order.
function _cfs_reduced_dms_su2(shells::Vector{_CFSShellSU2})
    N = length(shells)
    ρ = Vector{Dict{Tuple{Int,Rational{Int}},Vector{Float64}}}(undef, N)
    lst = shells[N]
    gE = minimum(minimum(lst.vals[qn][idx]) for (qn, idx) in lst.plan)
    ρ[N] = Dict(qn => zeros(length(idx)) for (qn, idx) in lst.plan)
    gstates = [
        (qn, p) for (qn, idx) in lst.plan for
        (p, i) in enumerate(idx) if isapprox(lst.vals[qn][i], gE; atol=1.0e-9)
    ]
    for (qn, p) in gstates
        ρ[N][qn][p] = 1.0 / length(gstates)
    end
    for n in (N - 1):-1:1
        child = shells[n + 1]
        ρn = Dict(qn => zeros(length(idx)) for (qn, idx) in shells[n].plan)
        for (nqn, idx) in child.plan
            Vk = child.vecs[nqn][:, idx]
            w = ρ[n + 1][nqn]
            for (Qk, Sk, q, s, r) in child.seg[nqn]
                haskey(ρn, (Qk, Sk)) || continue
                ρP = ρn[(Qk, Sk)]
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

# complete-basis poles (ω_phys, weight, 0.0) for the per-spin spectral function A_↑ = A_total/2.
# Wigner-Eckart multiplicity weights: ADD (ground (Q,S) → discarded (Q+1,Sp)) carries (2Sp+1)/(2S+1)
# from Σ_{σ,m,M′}|CG|² = 2Sp+1; REMOVE (ground (Q+1,Sp) → discarded (Q,S)) carries 1 (the CG sum is
# (2S_ground+1), cancelling the 1/(2S_ground+1) member-average). Pole energies are within-shell
# differences × the shell scale, exactly as `_cfs_poles`.
function _cfs_poles_su2(shells::Vector{_CFSShellSU2}, ρ)
    N = length(shells)
    poles = Tuple{Float64,Float64,Float64}[]
    for n in 1:N
        sh = shells[n]
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        final(qn, i) = n == N || !(i in get(kept, qn, Set{Int}()))   # last shell: all states final
        for (qn, idx) in sh.plan
            Q, S = qn
            for Sp in (S - 1 // 2, S + 1 // 2)
                Sp ≥ 0 || continue
                tgt = (Q + 1, Sp)
                haskey(sh.Ofull, (Q, S, Sp)) || continue
                Ob = sh.Ofull[(Q, S, Sp)]                            # ⟨(Q+1,Sp)‖d†‖(Q,S)⟩, full basis
                # ADD (ω>0): |G⟩ kept (Q,S) → discarded (Q+1,Sp)
                if haskey(sh.vals, tgt)
                    wf = 0.5 * (2 * float(Sp) + 1) / (2 * float(S) + 1)
                    for (kp, ki) in enumerate(idx)
                        w0 = ρ[n][qn][kp]
                        w0 < 1.0e-15 && continue
                        Ek = sh.vals[qn][ki]
                        for si in 1:length(sh.vals[tgt])
                            final(tgt, si) || continue
                            g = Ob[si, ki]
                            g == 0.0 && continue
                            push!(
                                poles, (sh.ω * (sh.vals[tgt][si] - Ek), wf * w0 * g^2, 0.0)
                            )
                        end
                    end
                end
                # REMOVE (ω<0): |G⟩ kept (Q+1,Sp) → discarded (Q,S)
                if haskey(sh.plan, tgt)
                    for (kp, ki) in enumerate(sh.plan[tgt])
                        w0 = ρ[n][tgt][kp]
                        w0 < 1.0e-15 && continue
                        Ek = sh.vals[tgt][ki]
                        for si in 1:length(sh.vals[qn])
                            final(qn, si) || continue
                            g = Ob[ki, si]
                            g == 0.0 && continue
                            push!(
                                poles, (sh.ω * (Ek - sh.vals[qn][si]), 0.5 * w0 * g^2, 0.0)
                            )
                        end
                    end
                end
            end
        end
    end
    return poles
end

function _cfs_poles_for(::U1SU2, model::AndersonModel, alg::NRGAlgorithm)
    shells = _cfs_collect_su2(model, alg)
    return _cfs_poles_su2(shells, _cfs_reduced_dms_su2(shells))
end
