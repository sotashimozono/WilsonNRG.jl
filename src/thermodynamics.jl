# ===========================================================================
#  Impurity thermodynamics from the NRG energy flow (Krishna-murthy, Wilkins &
#  Wilson 1980; Bulla, Costi & Pruschke, RMP 80, 395 (2008), §III.A.3).
#
#  The impurity contribution to an extensive quantity is the two-run difference
#      X_imp(T) = X_full(T) − X_bath(T),
#  the bath being the bare Wilson chain (no impurity). Both runs are evaluated at
#  the same shell scale (aligned so they end on the same Wilson site f_k), at
#  dimensionless inverse temperature β̄ via the rescaled spectrum.
#
#  IMPORTANT: use an `EnergyCut` truncation for thermodynamics. The impurity
#  doubles the full run's state count (spin), so a fixed `KeepN` resolves the bath
#  only ~half as well as the reference run and the subtraction undershoots the
#  exact plateaus. Keeping a fixed energy window makes both runs resolve the same
#  states, and the free-spin plateaus (T·χ_imp = 1/4, S_imp = ln2) reproduce.
# ===========================================================================

# Canonical thermodynamics of one shell from its (energy, label) levels at β̄. The label and the
# weighting are symmetry-dependent: for `U1U1` each entry is one state (label = 2Sz); for `U1SU2`
# each is a (2S+1)-fold multiplet (label = 2S, with ⟨Sz²⟩ summed over members = (2S+1)·S(S+1)/3).
function _shell_thermo(levels, β::Real, ::U1U1)
    Z = 0.0
    Eav = 0.0
    Sz2 = 0.0
    for (e, twoSz) in levels
        w = exp(-β * e)
        Z += w
        Eav += e * w
        Sz2 += (twoSz / 2)^2 * w
    end
    return (S=β * (Eav / Z) + log(Z), χT=Sz2 / Z)   # χT = ⟨Sz²⟩ (gμ_B=1, ⟨Sz⟩=0)
end
function _shell_thermo(levels, β::Real, ::U1SU2)
    Z = 0.0
    Eav = 0.0
    Sz2 = 0.0
    for (e, twoS) in levels
        S = twoS / 2
        w = (twoS + 1) * exp(-β * e)                # 2S+1 degenerate states in the multiplet
        Z += w
        Eav += e * w
        Sz2 += S * (S + 1) / 3 * w                  # Σ_{Sz=-S}^{S} Sz² = (2S+1)·S(S+1)/3
    end
    return (S=β * (Eav / Z) + log(Z), χT=Sz2 / Z)
end
function _shell_thermo(levels, β::Real, sym::AbstractSymmetry)
    return throw(EngineUnimplemented("thermodynamics not implemented for $(typeof(sym))"))
end

_free_site(::AndersonModel) = AndersonModel(; U=0.0, εd=0.0, Γ=0.0, D=1.0)

"""
    bath_reference(model, alg) -> (; levels, scale)

NRG flow of the bare Wilson chain (no impurity) — the reference subtracted to get
the impurity contribution. `levels[m]` ends on Wilson site `f_m` (one fewer site
than the full run at the matching shell scale).
"""
function bath_reference(model::AbstractImpurityModel, alg::NRGAlgorithm)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sym = alg.symmetry
    sqrtΛ = sqrt(alg.discretization.Λ)
    state = impurity_init(_free_site(model), sym, chain)        # f₀ as a free electron site
    levels = Vector{Vector{Tuple{Float64,Int}}}()
    scale = Float64[]
    for m in 0:(alg.nsites - 2)                                 # attach f₁ … f_{nsites-1}
        enl = add_site(
            state,
            sym;
            coupling=chain.hopping[m + 1],
            rescale=sqrtΛ,
            onsite=chain.onsite[m + 2],
        )
        diag = diagonalize_blocks(enl, sym)
        plan = truncation_plan(diag.vals, alg.truncation, sym)
        state = update_operators(diag, plan, sym)
        push!(levels, block_levels(state, sym))
        push!(scale, shell_scale(alg.discretization, m + 1))    # last site f_{m+1}
    end
    return (; levels, scale)
end

"""
    thermodynamics(model, alg; betabar = 1.0) -> (; T, S_imp, Tχ_imp)

Impurity entropy `S_imp(T)` and magnetic susceptibility `T·χ_imp(T)` across the
NRG flow, by the two-run (full − bath) subtraction. `T` is the shell temperature
`ωₙ/β̄`. For the symmetric Anderson model the curves show the textbook crossover

    free orbital  →  local moment    →  strong coupling (Kondo)
    S_imp:  ln4   →    ln2           →    0
    T·χ_imp: 1/8  →    1/4           →    0

Use `alg.truncation = EnergyCut(...)` (see the note in this file): a fixed `KeepN`
under-resolves the impurity-doubled run and the local-moment plateau undershoots.
"""
function thermodynamics(model::AbstractImpurityModel, alg::NRGAlgorithm; betabar::Real=1.0)
    sym = alg.symmetry                                  # selects the shell-sum (per-state vs multiplet)
    full = nrg_solve(model, alg)
    bath = bath_reference(model, alg)
    T = Float64[]
    S_imp = Float64[]
    Tχ_imp = Float64[]
    # Align full and bath on the same last Wilson site f_k. The bath always records from
    # f₁ (bath.levels[k] ends on f_k); the full run's first recorded shell ends on
    # f_{bath_sites_in_init(model)}, so full.levels[k+1-off] ends on f_k.
    off = bath_sites_in_init(model)
    for k in 1:(alg.nsites - 1)
        ft = _shell_thermo(full.levels[k + 1 - off], betabar, sym)
        bt = _shell_thermo(bath.levels[k], betabar, sym)
        push!(T, full.scale[k + 1 - off] / betabar)
        push!(S_imp, ft.S - bt.S)
        push!(Tχ_imp, ft.χT - bt.χT)
    end
    return (; T, S_imp, Tχ_imp)
end

# ⟨Sz⟩ of one shell in a field whose dimensionless Zeeman coefficient (β̄·h/ωₙ) multiplies Sz=D/2.
function _shell_mag(levels, β::Real, zeeman::Real, ::U1U1)
    Z = 0.0
    M = 0.0
    for (e, twoSz) in levels
        w = exp(-β * e + zeeman * (twoSz / 2))
        Z += w
        M += (twoSz / 2) * w
    end
    return M / Z
end
function _shell_mag(levels, β::Real, zeeman::Real, ::U1SU2)
    Z = 0.0
    M = 0.0
    for (e, twoS) in levels
        for twoSz in (-twoS):2:twoS                 # the field resolves the 2S+1 multiplet members
            Sz = twoSz / 2
            w = exp(-β * e + zeeman * Sz)
            Z += w
            M += Sz * w
        end
    end
    return M / Z
end
function _shell_mag(levels, β::Real, zeeman::Real, sym::AbstractSymmetry)
    return throw(EngineUnimplemented("magnetization not implemented for $(typeof(sym))"))
end

"""
    magnetization(model, alg; h, betabar = 1.0) -> (; T, M_imp)

Impurity magnetization `M_imp(T) = ⟨Sz⟩_full − ⟨Sz⟩_bath` in a uniform field `h`
(in units of the band half-width; `g μ_B = 1`), across the NRG flow. Because the
field `−h·Sz` commutes with `H` (Sz is conserved), it is applied *exactly* in the
Boltzmann weights of the zero-field spectrum — no re-diagonalization needed.

Limits (symmetric Anderson): a free local moment gives `M_imp → ½ tanh(h/2T)`
(saturating to `½` as `T→0`), and the linear response `∂M_imp/∂h|₀ = χ_imp`
reproduces the fluctuation susceptibility from [`thermodynamics`](@ref)
(fluctuation–dissipation). Use an `EnergyCut` truncation (see this file's note).
"""
function magnetization(
    model::AbstractImpurityModel, alg::NRGAlgorithm; h::Real, betabar::Real=1.0
)
    sym = alg.symmetry
    full = nrg_solve(model, alg)
    bath = bath_reference(model, alg)
    T = Float64[]
    M_imp = Float64[]
    off = bath_sites_in_init(model)                    # full/bath alignment (see thermodynamics)
    for k in 1:(alg.nsites - 1)
        ω = full.scale[k + 1 - off]                    # = bath.scale[k] (aligned on site f_k)
        zeeman = betabar * h / ω                       # β̄·(h/ωₙ), the dimensionless Zeeman shift
        mf = _shell_mag(full.levels[k + 1 - off], betabar, zeeman, sym)
        mb = _shell_mag(bath.levels[k], betabar, zeeman, sym)
        push!(T, ω / betabar)
        push!(M_imp, mf - mb)
    end
    return (; T, M_imp)
end

# median without a Statistics dependency
function _median(v)
    return (
        s=sort(v); n=length(s); iseven(n) ? (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2 : s[n ÷ 2 + 1]
    )
end

"""
    wilson_ratio(model::AndersonModel, alg; betabar=1.0, s_lo=0.03, s_hi=0.25) -> (; T, R, R_fp)

Wilson ratio `R_W(T)` and its strong-coupling fixed-point value `R_fp`, reproducing the Kondo-limit
`R = 2` (Andrei, PRL 45, 379 (1980); Krishna-murthy–Wilkins–Wilson, PRB 21, 1044 (1980); Hewson,
*The Kondo Problem*, 1993).

`R_W = (χ_imp/γ_imp) / (χ_imp/γ_imp)|_free` — the impurity spin susceptibility over the specific-heat
coefficient, normalised to `1` for the non-interacting (free resonant level, `U=0`) Fermi liquid.
For a Fermi liquid the impurity entropy is linear, `S_imp ≈ γ_imp·T`, so `γ_imp = S_imp/T` (the
RELIABLE entropy slope — the specific heat via energy fluctuations is swamped by the even/odd parity
artefact of the two-run subtraction). With `χ_imp = Tχ_imp/T`,

    R_W(T) = (Tχ_imp / S_imp)(model) / (Tχ_imp / S_imp)(free),

the `T` cancels, and the deep-low-`T` two-run breakdown (`S_imp → const`) is a common discretisation
artefact that CANCELS in the ratio (impurity and free runs share the chain). `R_fp` is the median
over the screened Fermi-liquid **plateau** `s_lo < S_imp < s_hi` — below the crossover (`S_imp <
s_hi`) and above the noise floor of the two-run subtraction (`S_imp > s_lo`); the deep tail where
`S_imp` collapses to a small constant is excluded (it scatters wildly). `U1U1`.

Checks: `R_W(U=0) = 1`; `R_fp → 2` in the Kondo limit (large `U/Γ`), reached and roughly universal
for `U/Γ ≳ 6`; the crossover `1 → 2` as `U/Γ` grows. The exact continuum `R = 2` is approached as
`Λ → 1` / with z-averaging; at `Λ = 2.5` the fixed-point value sits at `≈ 2.0–2.1`.
"""
function wilson_ratio(
    model::AndersonModel,
    alg::NRGAlgorithm;
    betabar::Real=1.0,
    s_lo::Real=0.03,
    s_hi::Real=0.25,
)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("wilson_ratio needs U1U1 (got $(typeof(alg.symmetry)))"))
    free = AndersonModel(; U=0.0, εd=0.0, Γ=model.Γ, D=model.D)   # non-interacting reference (R_W ≡ 1)
    tm = thermodynamics(model, alg; betabar)
    tf = thermodynamics(free, alg; betabar)
    R = (tm.Tχ_imp ./ tm.S_imp) ./ (tf.Tχ_imp ./ tf.S_imp)
    fl = [k for k in eachindex(tm.T) if s_lo < tm.S_imp[k] < s_hi]   # Fermi-liquid plateau
    R_fp = isempty(fl) ? NaN : _median(R[fl])
    return (; T=tm.T, R, R_fp)
end
