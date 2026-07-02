# ===========================================================================
#  Impurity self-energy Σ(ω) from the Green's function — two formulations to
#  compare (Axis 4b). Σ is notoriously method-sensitive, so the package makes the
#  choice explicit (dispatch) with a robust default ([`default_self_energy_method`]).
# ===========================================================================

"""
    hybridization_function(model::AndersonModel, ω) -> ComplexF64

Complex hybridization `Δ(ω) = Σ_k |V_k|²/(ω-ε_k+i0⁺)` of the flat band:
`Re Δ = (Γ/π)·ln|(ω+D)/(ω-D)|`, `Im Δ = -Γ` for `|ω|<D` (else 0). The
non-interacting Green's function is `G₀(ω) = 1/(ω - ε_d - Δ(ω))`.
"""
function hybridization_function(model::AndersonModel, ω::Real)
    Γ, D = model.Γ, model.D
    return complex((Γ / π) * log(abs((ω + D) / (ω - D))), abs(ω) < D ? -Γ : 0.0)
end

"""
    default_self_energy_method() -> AbstractSelfEnergyMethod

The robust default, [`SelfEnergyTrick`](@ref) (`Σ = U·F/G`; `Σ ∝ U`, errors cancel
in `F/G`). [`Dyson`](@ref) is offered for comparison.
"""
default_self_energy_method() = SelfEnergyTrick()

"""
    self_energy([method,] model, alg; via=default_self_energy_method(), b=0.6, window=0.7, ω=nothing, kw...) -> (; ω, Σ)

Impurity self-energy `Σ_σ(ω)`. `method` is the spectral method building `G` (default
`BHP`); `via` is how `Σ` is extracted: `SelfEnergyTrick()` (robust, `Σ=U·F/G`) or
`Dyson()` (`Σ=ω-ε_d-Δ-1/G`). At the symmetric point a Fermi liquid gives
`ReΣ(0)=U/2`, `ImΣ(0)=0`; `U=0 ⇒ Σ=0` (exact for the trick).

Dyson dispatches on `method` generically (any spectral method that yields `G`, so
`CFS`/`FDM` work — pass their parameters via `kw...`, e.g. `T` for `FDM`); the trick
needs the second correlator `F`, currently produced only by `BHP`.
"""
function self_energy(
    method::AbstractSpectralMethod,
    model::AndersonModel,
    alg::NRGAlgorithm;
    via::AbstractSelfEnergyMethod=default_self_energy_method(),
    b::Real=0.6,
    window::Real=0.7,
    ω=nothing,
    kw...,
)
    # Symmetry support is enforced PER PATH, not blanket: the trick's `_gf_poles` needs `U1U1`
    # (the BHP F-correlator indexes the (Q,2Sz,σ) blocks), while `Dyson` delegates to
    # `green_function`, which now also supports `U1SU2` (cfs_su2.jl) — so `Dyson` via `CFS`/`FDM`
    # yields the U1SU2 self-energy for free, and the trick still refuses non-U1U1 honestly.
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    return _self_energy(via, method, model, alg, ωs, b, window; kw...)
end
function self_energy(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return self_energy(default_spectral_method(), model, alg; kw...)
end

# The trick needs the F-correlator, currently produced only by BHP — encoded in DISPATCH
# (a (::SelfEnergyTrick, ::BHP) method + an AbstractSpectralMethod fallback that throws) rather
# than a runtime `isa` guard, so the precondition is a dispatch invariant.
function _self_energy(::SelfEnergyTrick, ::BHP, model, alg, ωs, b, window; kw...)
    poles = _trick_poles(alg.symmetry, model, alg, window)     # G/F windowed poles, symmetry-dispatched
    G = _correlator(poles, ωs, b, 2)
    F = _correlator(poles, ωs, b, 3)
    return (; ω=ωs, Σ=model.U .* F ./ G)
end
# windowed G/F poles for the trick: U1U1 sums the (Q,2Sz,σ) blocks; U1SU2 (spectral_su2.jl)
# propagates the impurity d† AND the compound O_F = n_↓ d†_↑ as spin-½ tensors (the CG weight
# cancels in F/G, so Σ=U·F/G stays exact — ReΣ(0)=U/2 — even though the windowed G alone is crude).
_trick_poles(::U1U1, model, alg, window) = _gf_poles(model, alg; window, with_F=true)
function _self_energy(
    ::SelfEnergyTrick, method::AbstractSpectralMethod, model, alg, ωs, b, window; kw...
)
    return throw(
        EngineUnimplemented(
            "the self-energy trick needs the F-correlator, currently produced only by BHP " *
            "(got $(typeof(method))); use via=Dyson() for a generic G-based self-energy",
        ),
    )
end

# magnitude-floored reciprocal: keeps 1/G finite where G→0 (band edges / spectral gaps),
# so the (admittedly noisy) Dyson self-energy never blows up to Inf on a user-supplied grid.
function _safe_invG(z::Complex, floor::Real)
    a = abs(z)
    a ≥ floor && return 1 / z
    a == 0 && return complex(1 / floor)
    return conj(z) / (a * floor)
end

# G from a computed Σ via the Dyson reconstruction G = 1/(ω−εd−Δ−Σ), denominator magnitude-floored
# (via _safe_invG) so it stays finite at band edges / spectral gaps. The SINGLE source of truth shared
# by improved_green_function and the impurity_solve seam (solver.jl) — keep the reconstruction here.
function _green_from_self_energy(model::AndersonModel, ω, Σ)
    Δ = hybridization_function.(Ref(model), ω)
    denom = @. ω - model.εd - Δ - Σ
    gfloor = 1.0e-12 * maximum(abs, denom)
    return _safe_invG.(denom, gfloor)
end
function _self_energy(
    ::Dyson, method::AbstractSpectralMethod, model, alg, ωs, b, window; kw...
)
    G = green_function(method, model, alg; b, ω=ωs, kw...).G        # any spectral method's G
    Δ = hybridization_function.(Ref(model), ωs)
    gfloor = 1.0e-12 * maximum(abs, G)
    return (;
        ω=ωs, Σ=[ωs[i] - model.εd - Δ[i] - _safe_invG(G[i], gfloor) for i in eachindex(ωs)]
    )
end

"""
    compare_self_energy(model, alg; method=default_spectral_method(), vias=(SelfEnergyTrick(), Dyson()), b=0.6, window=0.7, ω=nothing, kw...)
        -> (; ω, Σ, disagreement)

Run several self-energy formulations on a common grid and report `Σ` per method plus
the max pairwise `|ΔReΣ|` near `ω=0`. Cross-method agreement is the robustness signal
(at `U=0` the trick is exactly 0 while Dyson carries the broadening error — the gap is
the point). `method` is the spectral method building `G` (default `BHP`); for a non-BHP
`method` pass `vias=(Dyson(),)` (the trick is BHP-only) and any method kwargs (e.g. `T`).
"""
function compare_self_energy(
    model::AbstractImpurityModel,
    alg::NRGAlgorithm;
    method::AbstractSpectralMethod=default_spectral_method(),
    vias=(SelfEnergyTrick(), Dyson()),
    b::Real=0.6,
    window::Real=0.7,
    ω=nothing,
    kw...,
)
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    Σ = Dict(
        nameof(typeof(v)) =>
            self_energy(method, model, alg; via=v, b, window, ω=ωs, kw...).Σ for v in vias
    )
    near0 = findall(x -> abs(x) < 0.1, ωs)
    ks = collect(keys(Σ))
    dis = if isempty(near0) || length(ks) < 2
        0.0
    else
        maximum(maximum(abs, real.(Σ[a][near0] .- Σ[c][near0])) for a in ks, c in ks)
    end
    return (; ω=ωs, Σ, disagreement=dis)
end

"""
    improved_green_function([method,] model, alg; via=default_self_energy_method(), kw...) -> (; ω, G)

The self-energy-improved impurity Green's function `G(ω) = 1/(ω − εd − Δ(ω) − Σ(ω))`, with `Σ` from
[`self_energy`](@ref) and `Δ` the [`hybridization_function`](@ref) — the standard accurate NRG
spectral function (Bulla, Costi & Pruschke, RMP 80, 395 (2008), §III.B). Because the Fermi-liquid
pins `ReΣ(0)=U/2`, `ImΣ(0)=0` fix the `ω=0` self-energy, the Kondo resonance is tied to the UNITARY
LIMIT `πΓA(0) = sin²(πn_d/2) = 1` at the symmetric point — unlike the broadening-limited DIRECT
spectral function, whose `~T_K`-narrow Kondo peak the log-Gaussian washes out. `A(ω) = -Im G/π`.

NB the default `via=SelfEnergyTrick()` (`Σ = U·F/G`, from the separate F-correlator) is the
genuine self-energy correction; `via=Dyson()` here is an ALGEBRAIC IDENTITY — `Σ=ω-εd-Δ-1/G` fed
back gives `G = 1/(1/G)`, reproducing the input `G` to machine precision — so it is a no-op, not an
independent method (do not use it as a cross-method check).
"""
function improved_green_function(
    method::AbstractSpectralMethod,
    model::AndersonModel,
    alg::NRGAlgorithm;
    via::AbstractSelfEnergyMethod=default_self_energy_method(),
    kw...,
)
    se = self_energy(method, model, alg; via, kw...)
    return (; ω=se.ω, G=_green_from_self_energy(model, se.ω, se.Σ))
end
function improved_green_function(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return improved_green_function(default_spectral_method(), model, alg; kw...)
end
# clean refusal for unsupported models (mirrors green_function's generic fallback), not a MethodError
function improved_green_function(
    ::AbstractSpectralMethod, model::AbstractImpurityModel, ::NRGAlgorithm; kw...
)
    return throw(
        EngineUnimplemented(
            "improved_green_function on $(typeof(model)) not implemented; AndersonModel is available.",
        ),
    )
end
