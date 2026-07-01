# ===========================================================================
#  Impurity static properties — occupation ⟨n_d⟩ (KWW II, PRB 21, 1044 (1980)).
#
#  The T=0 impurity occupation per spin is the total *removal* (hole) spectral weight
#      ⟨n_{dσ}⟩ = ∫_{-∞}^{0} A_{dσ}(ω) dω = Σ_f |⟨f| d_σ |G⟩|² = ⟨G| d†_σ d_σ |G⟩,
#  which the complete-Fock-space basis (cfs.jl) sums *exactly* by completeness — the
#  same REMOVE channel as `_cfs_poles`, accumulating weight rather than binning poles,
#  so it is free of broadening error. This reproduces the asymmetric-Anderson static
#  properties of Krishna-murthy, Wilkins & Wilson II and, via the Friedel sum rule
#  (Langreth, PR 150, 516 (1966)), links the occupation to the ω=0 spectral pin.
# ===========================================================================

# total removal (hole) spectral weight for spin σ = ⟨n_{dσ}⟩ (T=0, exact by completeness).
# Mirrors the REMOVE block of `_cfs_poles` but sums weights instead of emitting poles.
function _cfs_hole_weight(shells::Vector{_CFSShell}, ρ, σ::Int)
    N = length(shells)
    nσ = 0.0
    for n in 1:N
        sh = shells[n]
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        final(qn, i) = n == N || !(i in get(kept, qn, Set{Int}()))   # last shell: all final
        for (qn, _) in sh.plan
            Q, D = qn
            tgt = (Q + 1, D + σ)
            (haskey(sh.Ofull, (Q, D, σ)) && haskey(sh.plan, tgt)) || continue
            Ob = sh.Ofull[(Q, D, σ)]                                 # ⟨tgt| d†_σ |qn⟩
            for (kp, ki) in enumerate(sh.plan[tgt])
                w0 = ρ[n][tgt][kp]                                   # ground weight on |Q+1,ki⟩
                w0 < 1.0e-15 && continue
                for s in 1:length(sh.vals[qn])                       # final |Q,s⟩ = d_σ|Q+1,ki⟩
                    final(qn, s) || continue
                    g = Ob[ki, s]
                    g == 0.0 && continue
                    nσ += w0 * g^2
                end
            end
        end
    end
    return nσ
end

"""
    occupation(model::AndersonModel, alg::NRGAlgorithm) -> (; total, up, dn)

T=0 impurity occupation `⟨n_d⟩ = ⟨n_{d↑}⟩ + ⟨n_{d↓}⟩` of the Anderson model, as the total
removal (hole) spectral weight summed over the complete Fock-space basis (`U1U1`; exact by
completeness, broadening-free). Reproduces the asymmetric-Anderson static properties of
Krishna-murthy, Wilkins & Wilson II (PRB 21, 1044 (1980)): `⟨n_d⟩ = 1` at the
particle–hole-symmetric point `εd = -U/2` for any `U`, and the empty-orbital → mixed-valence
→ Kondo crossover as `εd` sweeps. In the wide-band limit the U=0 value is
`⟨n_{dσ}⟩ = 1/2 - (1/π)·arctan(εd/Γ)`. See also the Friedel sum rule [`friedel_pin`](@ref).
"""
function occupation(model::AndersonModel, alg::NRGAlgorithm)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("occupation needs U1U1 (got $(typeof(alg.symmetry)))"))
    shells = _cfs_collect(model, alg)
    ρ = _cfs_reduced_dms(shells)
    up = _cfs_hole_weight(shells, ρ, 1)
    dn = _cfs_hole_weight(shells, ρ, -1)
    return (; total=up + dn, up=up, dn=dn)
end

# impurity double-occupancy operator n↑n↓ = projector on |↑↓⟩ = block (Q=2, D=0)
const _N_UP_DN = Dict((2, 0) => fill(1.0, 1, 1))

# ⟨G|O|G⟩ at the last-shell ground state, averaged over the (possibly degenerate) ground multiplet
function _ground_expectation(st::U1U1State, O)
    gE = minimum(minimum(v) for v in values(st.E))
    num = 0.0
    cnt = 0
    for (qn, ev) in st.E
        for (i, e) in enumerate(ev)
            isapprox(e, gE; atol=1.0e-9) || continue
            num += haskey(O, qn) ? O[qn][i, i] : 0.0        # 0 on blocks the observable misses
            cnt += 1
        end
    end
    return num / cnt
end

"""
    double_occupancy(model::AndersonModel, alg) -> Float64

T=0 impurity double occupancy `⟨n_{d↑} n_{d↓}⟩` of the Anderson model (`U1U1`) — the ground-state
weight on the doubly-occupied impurity, the charge-fluctuation static property of Krishna-murthy,
Wilkins & Wilson II (PRB 21, 1044 (1980)). Evaluated by propagating the `n↑n↓` observable along the
flow ([`propagate_observable`](@ref)) and reading `⟨G|n↑n↓|G⟩`. At `U=0` the spins are uncorrelated
so `⟨n_{d↑}n_{d↓}⟩ = ⟨n_{d↑}⟩⟨n_{d↓}⟩`; the Coulomb `U` suppresses it below that value. The local
charge fluctuation is `⟨n_d²⟩ - ⟨n_d⟩² = ⟨n_d⟩ + 2⟨n_{d↑}n_{d↓}⟩ - ⟨n_d⟩²` with [`occupation`](@ref).
"""
function double_occupancy(model::AndersonModel, alg::NRGAlgorithm)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("double_occupancy needs U1U1 (got $(typeof(alg.symmetry)))"))
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sqrtΛ = sqrt(alg.discretization.Λ)
    st = impurity_init(model, U1U1(), chain)
    O = deepcopy(_N_UP_DN)
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        plan = truncation_plan(diag.vals, alg.truncation, U1U1())
        O = propagate_observable(O, diag, plan, U1U1())
        st = update_operators(diag, plan, U1U1())
    end
    return _ground_expectation(st, O)
end
