# ===========================================================================
#  Dynamics from the impurity Green's function G(ω) (self-energy trick: Bulla–Hewson–Pruschke,
#  J. Phys.: Condens. Matter 10, 8365 (1998); review: Bulla–Costi–Pruschke, RMP 80, 395 (2008), §III.B).
#
#  Everything dynamical is derived from the complex retarded G(ω):
#      A(ω) = -Im G(ω)/π            ([`spectral`](@ref))
#      Σ(ω)  from G (+ a second correlator)   ([`self_energy`](@ref))
#  G is built by propagating d†_σ along the flow ([`propagate_operator`](@ref)),
#  collecting ground-state transitions in each shell's resolution WINDOW (rescaled
#  excitation in [w, w√Λ], so physical windows tile — kills the ×N_iter over-count),
#  broadening A with a log-Gaussian, and getting Re G by Kramers–Kronig.
#  The self-energy trick additionally tracks O₂ = n_{-σ} d†_σ for F = ⟨⟨d_σ n_{-σ}; d†_σ⟩⟩.
#  Per-spin convention: ∫A_σ dω = ⟨{d_σ,d†_σ}⟩ = 1.
# ===========================================================================

# log-Gaussian kernel; ∫dω of one pole returns its weight.
function _log_gaussian(ω::Real, ωp::Real, w::Real, b::Real)
    (ωp != 0 && ω != 0 && sign(ω) == sign(ωp)) || return 0.0
    return w / (b * sqrt(π) * abs(ω)) * exp(-(log(abs(ω / ωp)) / b)^2)
end

# n_{-σ} d†_σ for σ=↑ on the Anderson impurity: d†_↑ restricted to a ↓-occupied source (|↓⟩→|↑↓⟩).
_compound_operator(F0) = Dict(k => copy(v) for (k, v) in F0 if k == (1, -1, 1))

# Collect (ω_phys, w_G, w_F) poles of A_↑ from ground-state d†_↑ / d_↑ transitions, windowed.
# w_G = ⟨r|d†_↑|0⟩², w_F = ⟨r|n_↓d†_↑|0⟩·⟨r|d†_↑|0⟩ (0 when `with_F=false`).
function _gf_poles(
    model::AbstractImpurityModel, alg::NRGAlgorithm; window::Real, with_F::Bool
)
    alg.symmetry isa U1U1 || throw(
        EngineUnimplemented(
            "_gf_poles needs U1U1: it indexes the (Q, 2Sz, σ) operator blocks with Int arithmetic; " *
            "U1SU2 stores Rational-spin keys. (Callers already guard, but this is defensive.)",
        ),
    )
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sym = alg.symmetry
    sqrtΛ = sqrt(alg.discretization.Λ)
    whi = window * sqrtΛ
    st = impurity_init(model, sym, chain)
    O1 = deepcopy(st.F)                                  # d†_σ
    O2 = with_F ? _compound_operator(st.F) : Dict{NTuple{3,Int},Matrix{Float64}}()
    poles = Tuple{Float64,Float64,Float64}[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, sym; coupling, rescale, onsite=chain.onsite[n + 1]), sym
        )
        plan = truncation_plan(diag.vals, alg.truncation, sym)
        gqn = argmin(qn -> minimum(diag.vals[qn][plan[qn]]), collect(keys(plan)))
        i0 = plan[gqn][argmin(diag.vals[gqn][plan[gqn]])]
        E0 = diag.vals[gqn][i0]
        ωN = shell_scale(alg.discretization, n)
        O1 = propagate_operator(O1, diag, plan, sym)
        with_F && (O2 = propagate_operator(O2, diag, plan, sym))
        row = findfirst(==(i0), plan[gqn])
        addk = (gqn[1], gqn[2], 1)                        # d†_↑ : |0⟩ → (Q+1, D+1)  (ω>0)
        if haskey(O1, addk)
            tgt = (gqn[1] + 1, gqn[2] + 1)
            o2 = get(O2, addk, nothing)
            for (j, r) in enumerate(plan[tgt])
                x = diag.vals[tgt][r] - E0
                window ≤ x < whi || continue
                g = O1[addk][j, row]
                push!(poles, (x * ωN, g^2, o2 === nothing ? 0.0 : o2[j, row] * g))
            end
        end
        remk = (gqn[1] - 1, gqn[2] - 1, 1)                # d_↑ : |0⟩ → (Q−1, D−1)  (ω<0)
        if haskey(O1, remk)
            src = (gqn[1] - 1, gqn[2] - 1)
            o2 = get(O2, remk, nothing)
            for (j, r) in enumerate(plan[src])
                x = diag.vals[src][r] - E0
                window ≤ x < whi || continue
                g = O1[remk][row, j]
                push!(poles, (-x * ωN, g^2, o2 === nothing ? 0.0 : o2[row, j] * g))
            end
        end
        st = update_operators(diag, plan, sym)
    end
    return poles
end

# Kramers–Kronig: Re G(ωᵢ) = P∫dω' A(ω')/(ωᵢ-ω') ≈ Σ_{j≠i} A_j Δω_j /(ωᵢ-ω_j).
function _kramers_kronig(ωs::AbstractVector, A::AbstractVector)
    n = length(ωs)
    R = zeros(n)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            j == i && continue
            dω = (ωs[min(j + 1, n)] - ωs[max(j - 1, 1)]) / 2
            s += A[j] * dω / (ωs[i] - ωs[j])
        end
        R[i] = s
    end
    return R
end

# complex retarded correlator on `ωs` from broadened poles (column 2 = G weight, 3 = F weight)
function _correlator(poles, ωs, b, wcol)
    A = [sum(_log_gaussian(ω, p[1], p[wcol], b) for p in poles; init=0.0) for ω in ωs]
    return _kramers_kronig(ωs, A) .- im * π .* A
end

# default log-spaced ± frequency grid spanning the flow's scales
function _default_omega(model, alg; nω=240)
    D = hasproperty(model, :D) ? model.D : 1.0
    lo = D * alg.discretization.Λ^(-(alg.nsites) / 2) / 2
    hi = 2 * D
    pos = exp10.(range(log10(lo), log10(hi); length=nω))
    return vcat(-reverse(pos), pos)
end

"""
    default_spectral_method() -> AbstractSpectralMethod

The robust default spectral method, currently `BHP()`. The complete-basis [`CFS`](@ref)
(T=0, sum rule exact by completeness) is also available for comparison; the finite-T
`FDM()` becomes the default once it lands.
"""
default_spectral_method() = BHP()

"""
    green_function([method,] model, alg; b=0.6, window=0.7, ω=nothing) -> (; ω, G)

Retarded impurity Green's function `G_σ(ω)` (complex, per spin) under spectral
`method` (default [`default_spectral_method`](@ref)). `A(ω) = -Im G/π` is
[`spectral`](@ref); the self-energy follows via [`self_energy`](@ref).
"""
function green_function(
    ::BHP, model::AndersonModel, alg::NRGAlgorithm; b::Real=0.6, window::Real=0.7, ω=nothing
)
    alg.symmetry isa U1U1 || throw(
        EngineUnimplemented("BHP green_function needs U1U1 (got $(typeof(alg.symmetry)))"),
    )
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    poles = _gf_poles(model, alg; window, with_F=false)
    return (; ω=ωs, G=_correlator(poles, ωs, b, 2))
end
function green_function(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return green_function(default_spectral_method(), model, alg; kw...)
end
function green_function(
    method::AbstractSpectralMethod, ::AbstractImpurityModel, ::NRGAlgorithm; kw...
)
    return throw(
        EngineUnimplemented(
            "green_function via $(typeof(method)) not implemented; BHP is available."
        ),
    )
end

"""
    spectral([method,] model, alg; b=0.6, window=0.7, ω=nothing) -> (; ω, A)

Zero-temperature impurity spectral function `A_σ(ω) = -Im G_σ(ω)/π` (per spin).
`U = 0` recovers the resonant level `(Γ/π)/(ω²+Γ²)` ([`resonant_level_spectral`](@ref))
up to log-Gaussian broadening; `∫A dω = 1` and `A(ω) = A(−ω)` (symmetric point) hold.
"""
function spectral(
    method::AbstractSpectralMethod, model::AbstractImpurityModel, alg::NRGAlgorithm; kw...
)
    gf = green_function(method, model, alg; kw...)
    return (; ω=gf.ω, A=(-1 / π) .* imag.(gf.G))
end
function spectral(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return spectral(default_spectral_method(), model, alg; kw...)
end

"""
    spectral_at_zero(ω, A; window=0.1) -> Float64

Windowed spectral density at `ω=0`: the `dω`-weighted mean of `A(ω)` over `|ω| < window`
(≈ `∫_{|ω|<window} A dω / 2window`). Robust to the sharp z-interleaved peaks of a
[`zavg_spectral`](@ref) spectrum — a single grid point can fall in a valley between poles and
read ~0, whereas the window mean recovers the smooth physical `A(0)`. Pick `window` a fraction
of the lowest relevant scale (≈ `T_K/2` for the Kondo resonance). Use `πΓ·spectral_at_zero(...)`
for the Friedel unitary limit (`=1` at the symmetric point).
"""
function spectral_at_zero(ω::AbstractVector, A::AbstractVector; window::Real=0.1)
    num = 0.0
    den = 0.0
    @inbounds for k in 1:(length(ω) - 1)
        if abs(ω[k]) < window
            dω = ω[k + 1] - ω[k]
            num += A[k] * dω
            den += dω
        end
    end
    den > 0 || throw(ArgumentError("spectral_at_zero: no grid points within |ω| < $window"))
    return num / den
end
