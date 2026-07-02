# ===========================================================================
#  Self-energy TRICK (Σ = U·F/G) for the U1SU2 engine — the accurate self-energy, giving the
#  Fermi-liquid pin ReΣ(0)=U/2 at the symmetric point (which the broadening-limited Dyson route
#  does not). F = ⟨⟨d_↑ n_↓; d†_↑⟩⟩; the compound operator O_F = n_↓ d†_↑ is (exactly as in U1U1,
#  cf. `_compound_operator`) just d†_↑ with the (0,0)→(1,½) block dropped — still a spin-½ tensor,
#  so it propagates by `propagate_impurity_op` like d† itself. Both G and F are built from the SAME
#  windowed ground-state poles, so the per-spin Clebsch–Gordan weight cancels in F/G and Σ is robust
#  (this is why the standalone BHP spectral A(ω) is NOT exposed for U1SU2 — the windowed G alone
#  carries the ground-multiplet-averaging + BHP crudeness, but the F/G ratio is exact where it
#  matters). NB the SU(2) self-energy is used only via the trick; there is no green_function(::BHP,
#  ::U1SU2) — CFS (cfs_su2.jl) is the accurate U1SU2 spectral method.
# ===========================================================================

# per-spin-↑ Clebsch–Gordan weight for a ground multiplet Sg (averaged over its 2Sg+1 members)
# creating into a final multiplet Sp via d†_↑ (μ=+½).
function _su2_pup(Sg::Rational{Int}, Sp::Rational{Int})
    s = 0.0
    Mg = -Sg
    while Mg ≤ Sg
        c = clebsch_gordan(Sg, Mg, 1 // 2, 1 // 2, Sp, Mg + 1 // 2)
        s += c^2
        Mg += 1
    end
    return s / (2 * float(Sg) + 1)
end

# BHP-windowed G/F poles (ω_phys, w_G, w_F) for the U1SU2 self-energy trick. Mirrors `_gf_poles`
# (U1U1): per shell, the ground multiplet's windowed d†_↑ excitations (rescaled x ∈ [w, w√Λ]);
# w_G = P↑·⟨f‖d†‖G⟩², w_F = P↑·⟨f‖O_F‖G⟩·⟨f‖d†‖G⟩.
function _gf_poles_su2(
    model::AbstractImpurityModel, alg::NRGAlgorithm; window::Real, with_F::Bool
)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sqrtΛ = sqrt(alg.discretization.Λ)
    whi = window * sqrtΛ
    st = impurity_init(model, U1SU2(), chain)
    O1 = deepcopy(st.F)                                                    # d†_↑ (reduced)
    # O_F = n_↓ d†_↑ = d†_↑ with the (0,0)→(1,½) block dropped (spin-½ tensor, propagates like d†)
    O2 = if with_F
        Dict(k => copy(v) for (k, v) in st.F if k != (0, 0 // 1, 1 // 2))
    else
        Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}()
    end
    poles = Tuple{Float64,Float64,Float64}[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, U1SU2(); coupling, rescale, onsite=chain.onsite[n + 1]), U1SU2()
        )
        plan = truncation_plan(diag.vals, alg.truncation, U1SU2())
        gqn = argmin(qn -> minimum(diag.vals[qn][plan[qn]]), collect(keys(plan)))
        i0pos = argmin(diag.vals[gqn][plan[gqn]])
        E0 = diag.vals[gqn][plan[gqn][i0pos]]
        ωN = shell_scale(alg.discretization, n)
        O1k = _restrict_impurity_op(propagate_impurity_op(O1, diag), plan)
        O2k = if with_F
            _restrict_impurity_op(propagate_impurity_op(O2, diag), plan)
        else
            Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}()
        end
        Qg, Sg = gqn
        for Sp in (Sg - 1 // 2, Sg + 1 // 2)
            Sp ≥ 0 || continue
            P = _su2_pup(Sg, Sp)
            # ADD (ω>0): ground (Qg,Sg) → kept (Qg+1, Sp)
            tgt = (Qg + 1, Sp)
            if haskey(O1k, (Qg, Sg, Sp)) && haskey(plan, tgt)
                g1 = O1k[(Qg, Sg, Sp)]
                g2 = get(O2k, (Qg, Sg, Sp), nothing)
                for (jp, r) in enumerate(plan[tgt])
                    x = diag.vals[tgt][r] - E0
                    window ≤ x < whi || continue
                    gG = g1[jp, i0pos]
                    push!(
                        poles,
                        (x * ωN, P * gG^2, g2 === nothing ? 0.0 : P * g2[jp, i0pos] * gG),
                    )
                end
            end
            # REMOVE (ω<0): ground (Qg,Sg) is the upper (Q+1,Sp) side; final (Qg−1, Sp) kept
            src = (Qg - 1, Sp)
            if haskey(O1k, (Qg - 1, Sp, Sg)) && haskey(plan, src)
                g1 = O1k[(Qg - 1, Sp, Sg)]
                g2 = get(O2k, (Qg - 1, Sp, Sg), nothing)
                for (jp, r) in enumerate(plan[src])
                    x = diag.vals[src][r] - E0
                    window ≤ x < whi || continue
                    gG = g1[i0pos, jp]
                    push!(
                        poles,
                        (-x * ωN, P * gG^2, g2 === nothing ? 0.0 : P * g2[i0pos, jp] * gG),
                    )
                end
            end
        end
        O1 = O1k
        O2 = O2k
        st = update_operators(diag, plan, U1SU2())
    end
    return poles
end

# the self-energy trick's windowed poles, dispatched on symmetry (U1U1 → spectral.jl `_gf_poles`).
_trick_poles(::U1SU2, model, alg, window) = _gf_poles_su2(model, alg; window, with_F=true)
