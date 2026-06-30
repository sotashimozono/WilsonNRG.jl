# ===========================================================================
#  The generic NRG driver and its engine seam.
#
#  `nrg_solve` is the model-/symmetry-agnostic scheme. It calls a small set of
#  seam functions that the symmetry layer implements — exactly the split the
#  `NonHermitianNRG` reference proves is possible but never abstracted (there the
#  loop, truncation, symmetry-enforcement and operator-update are generic, while
#  only the initial Hamiltonian, the chain parameters and the single-site
#  operator matrices are model-specific).
#
#  The deterministic seams (`wilson_chain`, `truncate_spectrum`) are implemented.
#  The many-body seams (`impurity_init`, `add_site`, `diagonalize_blocks`,
#  `update_operators`) are declared here and land per the staged roadmap, each
#  behind its own faithfulness gate. Calling the engine before they exist raises
#  a clear [`EngineUnimplemented`](@ref) rather than a cryptic `MethodError`.
# ===========================================================================

"""
    EngineUnimplemented(msg)

Raised when a symmetry/model combination's iterative engine has not been
implemented yet. The Wilson chain ([`wilson_chain`](@ref)) and the U=0 bootstrap
are always available.
"""
struct EngineUnimplemented <: Exception
    msg::String
end
Base.showerror(io::IO, e::EngineUnimplemented) = print(io, "EngineUnimplemented: ", e.msg)

# ---- engine seam (implemented per symmetry, Stage 1+) ---------------------

"""
    impurity_init(model, sym, chain) -> state

Build the iteration-0 state (impurity ⊗ first bath site) in the block structure
of `sym`: the block-diagonal Hamiltonian and the single-site fermion operators
to be propagated. Implemented per symmetry from Stage 1.
"""
function impurity_init end

"""
    add_site(state, chain, n, sym) -> state

Couple Wilson site `n` onto the kept space (the generic block-H assembly:
`ξₙ`-hopping between the stored operators of adjacent shells).
"""
function add_site end

"""
    diagonalize_blocks(state, sym) -> spectrum

Diagonalize each symmetry block of the enlarged Hamiltonian.
"""
function diagonalize_blocks end

"""
    update_operators(state, kept, sym) -> state

Project the tracked fermion operators onto the kept eigenbasis (the `UM`/`UMd`
update of the reference engine).
"""
function update_operators end

# Friendly fallbacks: the engine is not wired for any symmetry yet.
const _STAGE1 =
    "iterative engine: roadmap Stage 1 (U(1)×U(1) Anderson). " *
    "Available now: wilson_chain(...) and the U=0 bootstrap " *
    "(resonant_level_spectral / friedel_pin / spectral_sum_rule)."
function impurity_init(m::AbstractImpurityModel, s::AbstractSymmetry, ::WilsonChain)
    throw(
        EngineUnimplemented("impurity_init for $(typeof(m)) under $(typeof(s)) — $_STAGE1")
    )
end

# ---- truncation (generic, symmetry-agnostic) ------------------------------

"""
    truncate_spectrum(energies, trunc::AbstractTruncation) -> keep::Vector{Int}

Indices of the states to retain, given the iteration's rescaled `energies`.
Symmetry-agnostic; the block bookkeeping is the symmetry layer's job.
"""
function truncate_spectrum(energies::AbstractVector{<:Real}, trunc::KeepN)
    order = sortperm(energies)
    return order[1:min(trunc.N, length(order))]
end
function truncate_spectrum(energies::AbstractVector{<:Real}, trunc::EnergyCut)
    return findall(≤(trunc.Ecut), energies)
end

# ---- the scheme ------------------------------------------------------------

"""
    nrg_solve(model::AbstractImpurityModel, alg::NRGAlgorithm) -> NRGResult

Run the generic NRG scheme: discretize the bath, then iteratively enlarge,
diagonalize, rescale by `√Λ`, and truncate.

```
chain = wilson_chain(alg.discretization, model, alg.nsites)
state = impurity_init(model, alg.symmetry, chain)
for n in 1:alg.nsites
    state    = add_site(state, chain, n, alg.symmetry)
    spectrum = diagonalize_blocks(state, alg.symmetry)
    keep     = truncate_spectrum(spectrum, alg.truncation)
    state    = update_operators(state, keep, alg.symmetry)   # + √Λ rescale
end
```

The deterministic steps run today; the many-body seam raises
[`EngineUnimplemented`](@ref) until the symmetry layer lands (see the roadmap).
"""
function nrg_solve(model::AbstractImpurityModel, alg::NRGAlgorithm)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    _state = impurity_init(model, alg.symmetry, chain)   # Stage 1+
    return _state # unreachable until the engine is wired; shape is NRGResult
end

# ---- spectral layer (Axis 4) ----------------------------------------------

"""
    spectral(method::AbstractSpectralMethod, result::NRGResult, op) -> (ω, A)

Impurity spectral function from an NRG flow under formulation `method`
(Axis 4). Implemented per method from Stage 3 (`FDM` first).
"""
function spectral end
function spectral(method::AbstractSpectralMethod, ::NRGResult, op)
    throw(
        EngineUnimplemented("spectral via $(typeof(method)) — roadmap Stage 3 (FDM first).")
    )
end
