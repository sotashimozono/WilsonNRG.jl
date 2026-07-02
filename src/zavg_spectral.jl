# ===========================================================================
#  z-averaging of the impurity spectral function (Oliveira & Oliveira, PRB 49, 11986 (1994);
#  Žitko & Pruschke, PRB 79, 085106 (2009)). A single logarithmic discretization pins the spectral
#  poles to the grid `ωₙ ∝ Λ^{-(n-z)/2}`; averaging `A(ω)` over `nz` twist parameters `z ∈ (0,1]`
#  INTERLEAVES those grids, so the discretization + log-Gaussian broadening artifacts of any one `z`
#  cancel and `A(ω)` converges toward the exact continuum spectrum (e.g. the U=0 resonant-level
#  Lorentzian `(Γ/π)/(ω²+Γ²)`). Needs a z-capable discretization (`ZitkoPruschke` / `CampoOliveira`)
#  whose z-dependent `shell_scale` (discretization.jl) places each z's poles consistently — the
#  single-z schemes share one shell ladder, so this z-dependent scale is what makes the average
#  actually interleave rather than pile up.
# ===========================================================================

# rebuild a z-capable discretization at a shifted twist z (others refuse — not a bare MethodError)
_with_z(d::ZitkoPruschke, z::Real) = ZitkoPruschke(d.Λ; z)
_with_z(d::CampoOliveira, z::Real) = CampoOliveira(d.Λ; z)
function _with_z(d::AbstractDiscretization, ::Real)
    return throw(
        EngineUnimplemented(
            "z-averaging needs a z-capable discretization (ZitkoPruschke or CampoOliveira); " *
            "got $(typeof(d))",
        ),
    )
end

"""
    zavg_green_function([method,] model, alg; nz=16, ω=nothing, kw...) -> (; ω, G)

z-averaged retarded impurity Green's function (Oliveira–Oliveira 1994; Žitko–Pruschke, PRB 79,
085106 (2009)): the average of `green_function(method, model, …)` over `nz` discretization twists
`z ∈ {k/nz : k=1..nz}`. The log-grid poles of different `z` interleave, so the artifacts of any
single `z` cancel and `A(ω) = -Im G/π` converges toward the exact spectrum faster than any single
`z`. Requires a z-capable discretization (`ZitkoPruschke` / `CampoOliveira`); `kw...` (e.g. `b`,
`window`, `T`) pass through to the underlying `method`.
"""
function zavg_green_function(
    method::AbstractSpectralMethod,
    model::AbstractImpurityModel,
    alg::NRGAlgorithm;
    nz::Integer=16,
    ω=nothing,
    kw...,
)
    nz ≥ 1 || throw(ArgumentError("zavg_green_function: nz must be ≥ 1 (got $nz)"))
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    G = zeros(ComplexF64, length(ωs))
    for k in 1:nz
        alg_z = NRGAlgorithm(;
            discretization=_with_z(alg.discretization, k / nz),
            symmetry=alg.symmetry,
            truncation=alg.truncation,
            nsites=alg.nsites,
        )
        G .+= green_function(method, model, alg_z; ω=ωs, kw...).G
    end
    return (; ω=ωs, G=G ./ nz)
end
function zavg_green_function(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return zavg_green_function(default_spectral_method(), model, alg; kw...)
end

"""
    zavg_spectral([method,] model, alg; nz=16, ω=nothing, kw...) -> (; ω, A)

z-averaged impurity spectral function `A(ω) = -Im G/π` from [`zavg_green_function`](@ref).
"""
function zavg_spectral(
    method::AbstractSpectralMethod, model::AbstractImpurityModel, alg::NRGAlgorithm; kw...
)
    gf = zavg_green_function(method, model, alg; kw...)
    return (; ω=gf.ω, A=(-1 / π) .* imag.(gf.G))
end
function zavg_spectral(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    return zavg_spectral(default_spectral_method(), model, alg; kw...)
end
