# ===========================================================================
#  z-averaging / improved discretization (Žitko & Pruschke, PRB 79, 085106 (2009)).
#
#  The logarithmic discretization replaces each interval Iⱼ of the conduction band by one
#  representative level Eⱼ(z) (z = the grid "twist"); the z-averaged local DOS at the first
#  Wilson site is, exactly (Žitko Eq. 31),
#       A_{f0}(ω) = [∫_{Iⱼ} ρ dε] / |dEⱼ/dz|,   with z, j fixed by Eⱼ(z) = ω.
#  A faithful discretization must reproduce the band, A_{f0}(ω) = ρ(ω). The conventional
#  (Eq. 33, arithmetic mean) and Campo–Oliveira (Eq. 32, log mean) representative energies both
#  leave band-edge artefacts; the Žitko–Pruschke choice (Eq. 35) is constructed so A_{f0}=ρ holds,
#  which for the flat band modifies only the first interval (Eq. 36):
#       E₁(z) = (1 − Λ^{−z})/ln Λ + 1 − z.
# ===========================================================================

using LinearAlgebra: norm, dot

# z-shifted intervals of the (positive) flat band [0,1] in units of D, for twist z ∈ (0,1]:
#   I₁ = [Λ^{-z}, 1];  Iⱼ = [Λ^{-(j-1+z)}, Λ^{-(j-2+z)}]  (j ≥ 2)
_zavg_lo(Λ, j, z) = j == 1 ? Λ^(-z) : Λ^(-(j - 1 + z))
_zavg_hi(Λ, j, z) = j == 1 ? 1.0 : Λ^(-(j - 2 + z))

# representative energies Eⱼ(z) for the three schemes (flat band)
_rep_conventional(Λ, j, z) = (_zavg_lo(Λ, j, z) + _zavg_hi(Λ, j, z)) / 2          # Eq. 33
function _rep_campo_oliveira(Λ, j, z)                                              # Eq. 32 (log mean)
    a, b = _zavg_lo(Λ, j, z), _zavg_hi(Λ, j, z)
    return (b - a) / log(b / a)
end
_rep_zitko(Λ, j, z) =                                                              # Eq. 36 (j=1) else Eq. 32
    j == 1 ? (1 - Λ^(-z)) / log(Λ) + 1 - z : _rep_campo_oliveira(Λ, j, z)

const _ZAVG_SCHEMES = (
    conventional=_rep_conventional, campo_oliveira=_rep_campo_oliveira, zitko=_rep_zitko
)

"""
    band_dos(disc, model; scheme=:zitko, nint=12, nz=64) -> (; ω, A)

The z-averaged local density of states `A_{f0}(ω)` of the discretized **flat** conduction band
(Žitko–Pruschke Eq. 31), sampled parametrically over the twist `z`. A faithful discretization
reproduces the band, `A_{f0}(ω) = ρ(ω) = 1/(2D)`. `scheme ∈ (:conventional, :campo_oliveira,
:zitko)` selects the representative-energy recipe; `:zitko` (Eq. 35/36) is exact for the flat band.
"""
function band_dos(disc::AbstractDiscretization, model::AndersonModel; scheme::Symbol=:zitko,
    nint::Integer=12, nz::Integer=64)
    Λ, D = disc.Λ, model.D
    Efn = _ZAVG_SCHEMES[scheme]
    ωs = Float64[]
    As = Float64[]
    h = 1.0e-6
    for j in 1:nint, k in 1:nz
        z = k / nz                                   # z ∈ (0,1]
        ω = Efn(Λ, j, z)
        (0 < ω ≤ 1) || continue
        dEdz = (Efn(Λ, j, min(z + h, 1.0)) - Efn(Λ, j, max(z - h, 1.0e-9))) /
               (min(z + h, 1.0) - max(z - h, 1.0e-9))
        dEdz == 0 && continue
        weight = (_zavg_hi(Λ, j, z) - _zavg_lo(Λ, j, z)) / 2      # dimensionless ∫ρdε (ρ̃=1/2)
        push!(ωs, ω * D)
        push!(As, weight / abs(dEdz) / D)            # A_{f0}(ω) in 1/energy ⇒ ρ(ω)=1/(2D)
    end
    p = sortperm(ωs)
    return (; ω=ωs[p], A=As[p])
end

# z-shifted Wilson chain via Lanczos: the discretized-band star (both ± branches, representative
# energy `Efn`) is tridiagonalized from the f₀ coupling vector. The stored hopping is dimensionless,
# ξₙ = tₙ·Λ^{n/2} (the driver's √Λ recursion restores the physical tₙ ∝ Λ^{-n/2}); onsite = 0 by
# particle–hole symmetry. Same convention as the WilsonLog closed form, so it runs in `nrg_solve`.
function _zshift_chain(disc, Efn, model::AbstractImpurityModel, nsites::Integer)
    Λ, z, D = disc.Λ, disc.z, model.D                     # band half-width only; see below re: Γ
    εs = Float64[]
    γs = Float64[]
    for j in 1:(nsites + 40)                              # enough intervals for a stable chain
        a, c = D * _zavg_lo(Λ, j, z), D * _zavg_hi(Λ, j, z)
        c - a > 1.0e-14 * D || continue
        Erep = D * Efn(Λ, j, z)
        w2 = c - a                                        # ∝ ∫_{Iⱼ} ρ dε; the hybridization scale
        for s in (1.0, -1.0)                              # (Γ/π for Anderson) cancels in v=γs/‖γs‖,
            push!(εs, s * Erep)                           # so the chain is model-agnostic — it also
            push!(γs, sqrt(w2))                           # serves the Kondo band (no Γ field)
        end
    end
    onsite = zeros(nsites)
    hopping = zeros(nsites)
    vprev = zeros(length(εs))
    v = γs ./ norm(γs)                                    # f₀ ∝ Σ γⱼ aⱼ
    βprev = 0.0
    for n in 1:nsites
        Hv = εs .* v
        α = dot(v, Hv)                                    # ≈ 0 (±-symmetric band)
        w = Hv .- α .* v .- βprev .* vprev
        β = norm(w)
        n < nsites && (hopping[n] = β * Λ^(n / 2))        # ξₙ = tₙ·Λ^{n/2}
        β < 1.0e-13 && break
        vprev = v
        v = w ./ β
        βprev = β
    end
    return WilsonChain(onsite, hopping, disc)
end

"""
    wilson_chain(disc::CampoOliveira, model, nsites)
    wilson_chain(disc::ZitkoPruschke, model, nsites)

z-shifted Wilson chain for the flat band, via Lanczos of the discretized-band star (twist `z`
carried by `disc`). `CampoOliveira` uses the log-mean representative energy (Eq. 32);
`ZitkoPruschke` the artefact-free choice (Eq. 35/36). Both feed [`nrg_solve`](@ref) directly.
"""
wilson_chain(disc::CampoOliveira, model::AbstractImpurityModel, nsites::Integer) =
    _zshift_chain(disc, _rep_campo_oliveira, model, nsites)
wilson_chain(disc::ZitkoPruschke, model::AbstractImpurityModel, nsites::Integer) =
    _zshift_chain(disc, _rep_zitko, model, nsites)

# clean failure for a discretization without a wilson_chain (not a bare MethodError)
function wilson_chain(disc::AbstractDiscretization, ::AbstractImpurityModel, ::Integer)
    throw(EngineUnimplemented("wilson_chain not implemented for $(typeof(disc))"))
end
