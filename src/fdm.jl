# ===========================================================================
#  Full-density-matrix (FDM) spectral function at finite T — Weichselbaum & von
#  Delft, PRL 99, 076402 (2007). Generalizes the T=0 complete-basis CFS (cfs.jl):
#  the full thermal density matrix weights each discarded state |s⟩ₙ by
#  4^{N-n} e^{-βE_s} (environment degeneracy of the not-yet-added sites × Boltzmann),
#  so the reduced density matrices ρ^red_n are seeded thermally rather than by the
#  ground state alone, and the Lehmann sum runs over all *not-both-kept* pairs with
#  weight (ρ_i + ρ_j). The spectral sum rule ∫A_σ dω = 1 holds at any T (completeness).
#  At T=0 it reduces to CFS, to which `green_function(::FDM, …; T=0)` delegates.
#
#  Absolute many-body energies (needed for the Boltzmann weights — unlike CFS, whose
#  T=0 poles are within-shell differences) telescope across the √Λ rescaling:
#      E_abs[n][i] = ωₙ·vals_n[i] + Σ_{m<n} ωₘ·gₘ      (gₘ = ground subtracted at shell m),
#  consistent because ωₙ√Λ = ωₙ₋₁.
#
#  Near ω=0 the log-Gaussian kernel (∝1/|ω|) would diverge on the dense finite-T
#  quasi-elastic poles, so poles with |ωp| < ω₀ are broadened with a linear Gaussian of
#  width b·ω₀ (ω₀ ~ T, the thermal resolution); |ωp| ≥ ω₀ keep the log-Gaussian.
# ===========================================================================

# two-regime broadening kernel: log-Gaussian above ω₀, linear Gaussian below.
function _fdm_kernel(ω::Real, ωp::Real, w::Real, b::Real, ω0::Real)
    abs(ωp) ≥ ω0 && return _log_gaussian(ω, ωp, w, b)
    γ = b * ω0
    return w / (sqrt(2π) * γ) * exp(-(ω - ωp)^2 / (2γ^2))
end

# absolute many-body energies per shell on a common scale (for the Boltzmann weights).
function _fdm_abs_energies(shells::Vector{_CFSShell})
    Eabs = Vector{Dict{NTuple{2,Int},Vector{Float64}}}(undef, length(shells))
    refcum = 0.0
    for (n, sh) in enumerate(shells)
        Eabs[n] = Dict(qn => sh.ω .* v .+ refcum for (qn, v) in sh.vals)
        gn = minimum(minimum(sh.vals[qn][idx]) for (qn, idx) in sh.plan)
        refcum += sh.ω * gn
    end
    return Eabs
end

# log thermal weights on the terminal states (discarded at n<N, all states at the last
# shell) + the log partition function (numerically stable via the env-degeneracy 4^{N-n}).
function _fdm_log_weights(shells::Vector{_CFSShell}, Eabs, β::Real)
    N = length(shells)
    ln4 = log(4.0)
    Emin = minimum(minimum(v) for d in Eabs for v in values(d))
    logw = Vector{Dict{NTuple{2,Int},Vector{Float64}}}(undef, N)
    terms = Float64[]
    for (n, sh) in enumerate(shells)
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        envln = (N - n) * ln4
        lw = Dict{NTuple{2,Int},Vector{Float64}}()
        for (qn, v) in Eabs[n]
            arr = fill(-Inf, length(v))
            for i in eachindex(v)
                ((n == N) || !(i in get(kept, qn, Set{Int}()))) || continue
                arr[i] = envln - β * (v[i] - Emin)
                push!(terms, arr[i])
            end
            lw[qn] = arr
        end
        logw[n] = lw
    end
    M = maximum(terms)
    logZ = M + log(sum(exp(t - M) for t in terms))
    return logw, logZ
end

# thermally-seeded backward DM sweep → ρ^red_n on the full shell-n space (kept ← propagated,
# discarded ← terminal thermal weight). Reduces to the CFS ground-state sweep as T→0.
function _fdm_reduced_dms(shells::Vector{_CFSShell}, logw, logZ)
    N = length(shells)
    ρ = Vector{Dict{NTuple{2,Int},Vector{Float64}}}(undef, N)
    ρ[N] = Dict(qn => [isfinite(x) ? exp(x - logZ) : 0.0 for x in lw] for (qn, lw) in logw[N])
    for n in (N - 1):-1:1
        sh = shells[n]
        child = shells[n + 1]
        ρK = Dict(P => zeros(length(idx)) for (P, idx) in sh.plan)   # kept-position indexed
        for (nqn, V) in child.vecs
            haskey(ρ[n + 1], nqn) || continue          # vals/vecs share keys; guard the invariant
            ρc = ρ[n + 1][nqn]
            for (P, s, r) in child.seg[nqn]
                haskey(ρK, P) || continue
                ρKP = ρK[P]
                for p in eachindex(r)
                    row = r[p]
                    acc = 0.0
                    @inbounds for a in eachindex(ρc)
                        ρc[a] == 0.0 && continue
                        acc += abs2(V[row, a]) * ρc[a]
                    end
                    ρKP[p] += acc
                end
            end
        end
        ρn = Dict{NTuple{2,Int},Vector{Float64}}()
        for (qn, v) in sh.vals
            full = [isfinite(x) ? exp(x - logZ) : 0.0 for x in logw[n][qn]]   # w^D on discarded
            if haskey(sh.plan, qn)
                for (p, i) in enumerate(sh.plan[qn])
                    full[i] = ρK[qn][p]                                       # ρ^K on kept
                end
            end
            ρn[qn] = full
        end
        ρ[n] = ρn
    end
    return ρ
end

# unified not-both-kept poles (ω_phys, weight) for spin σ, Lehmann weight (ρ_i+ρ_j).
function _fdm_poles(shells::Vector{_CFSShell}, ρ, σ::Int; tol::Real=1e-15)
    N = length(shells)
    poles = Tuple{Float64,Float64}[]
    for n in 1:N
        sh = shells[n]
        kept = Dict(qn => Set(idx) for (qn, idx) in sh.plan)
        for ((Q, D, σd), Ob) in sh.Ofull
            σd == σ || continue
            src = (Q, D)
            tgt = (Q + 1, D + σ)
            (haskey(ρ[n], src) && haskey(ρ[n], tgt)) || continue
            ks = get(kept, src, Set{Int}())
            kt = get(kept, tgt, Set{Int}())
            ρs = ρ[n][src]
            ρt = ρ[n][tgt]
            Es = sh.vals[src]
            Et = sh.vals[tgt]
            for i in eachindex(Es)
                ri = ρs[i]
                ik = i in ks
                for j in eachindex(Et)
                    (n < N && ik && (j in kt)) && continue   # both kept ⇒ deferred to later shell
                    w = ri + ρt[j]
                    w < tol && continue
                    g = Ob[j, i]
                    g == 0.0 && continue
                    push!(poles, (sh.ω * (Et[j] - Es[i]), w * g^2))
                end
            end
        end
    end
    return poles
end

# complex retarded correlator from FDM poles with the two-regime kernel + Kramers-Kronig.
function _fdm_correlator(poles, ωs, b, ω0)
    A = [sum(_fdm_kernel(ω, p[1], p[2], b, ω0) for p in poles; init=0.0) for ω in ωs]
    return _kramers_kronig(ωs, A) .- im * π .* A
end

"""
    green_function(::FDM, model::AndersonModel, alg; T=0.0, b=0.6, ω=nothing, ω0=nothing) -> (; ω, G)

Retarded impurity Green's function via the full-density-matrix (FDM) method at
temperature `T` (Weichselbaum & von Delft, PRL 99, 076402 (2007)). The complete basis
of discarded states is weighted by the full thermal density matrix, so the spectral sum
rule `∫A_σ dω = 1` holds at any `T` (completeness). `T=0` delegates to the complete-basis
[`CFS`](@ref) (the exact ground-state projector). Near `ω=0` the resolution is set by the
two-regime crossover `ω0` (default `max(T, 3·ω_min)`, floored at the grid resolution so
the quasi-elastic weight is never broadened below the grid spacing and silently lost).
Needs `U1U1`. `A(ω) = -Im G/π` is [`spectral`](@ref).
"""
function green_function(
    ::FDM, model::AndersonModel, alg::NRGAlgorithm;
    T::Real=0.0, b::Real=0.6, ω=nothing, ω0=nothing,
)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("FDM green_function needs U1U1 (got $(typeof(alg.symmetry)))"))
    T ≥ 0 || throw(ArgumentError("FDM: temperature T must be ≥ 0 (got $T)"))
    T == 0 && return green_function(CFS(), model, alg; b, ω)         # T=0 ≡ CFS (exact GS projector)
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    # crossover ω₀ ~ T, floored at the grid resolution: if T is below the lowest grid scale
    # the linear-Gaussian quasi-elastic weight would fall between grid points and vanish silently.
    ωmin = model.D * alg.discretization.Λ^(-alg.nsites / 2) / 2
    ω0eff = ω0 === nothing ? max(T, 3 * ωmin) : ω0
    shells = _cfs_collect(model, alg)
    Eabs = _fdm_abs_energies(shells)
    logw, logZ = _fdm_log_weights(shells, Eabs, 1 / T)
    ρ = _fdm_reduced_dms(shells, logw, logZ)
    poles = _fdm_poles(shells, ρ, 1)                                 # spin ↑ (per-spin A)
    return (; ω=ωs, G=_fdm_correlator(poles, ωs, b, ω0eff))
end
