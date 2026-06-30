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
#  The deterministic seams (`wilson_chain`, `truncation_plan`) and the many-body
#  seams (`impurity_init`, `add_site`, `diagonalize_blocks`, `update_operators`)
#  are implemented for `U1U1`; symmetries without a wired layer raise a clear
#  [`EngineUnimplemented`](@ref) rather than a cryptic `MethodError`.
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

# ---- truncation (symmetry-aware via the `multiplicity` hook) --------------

"""
    multiplicity(sym::AbstractSymmetry, qn) -> Int

Number of physical states represented by one kept eigenvalue of block `qn`. For
abelian symmetries each `(Q, Sₙ)` state counts once (`1`); a non-abelian symmetry
overrides this with the multiplet dimension (e.g. `2S+1`) so a `KeepN` budget
counts physical states. This is the hook that lets [`truncation_plan`](@ref)
extend to SU(2)-graded blocks unchanged.
"""
multiplicity(::AbstractSymmetry, qn) = 1

"""
    truncation_plan(vals::Dict{K,Vector{Float64}}, trunc, sym) -> Dict{K,Vector{Int}}

Choose which states to keep *across* symmetry blocks from each block's energies.

- **Degeneracy-aware**: never splits a (near-)degenerate cluster, so exact
  degeneracies (e.g. spin-flip partners) and spectral sum rules survive truncation.
- **Multiplicity-weighted** via [`multiplicity`](@ref): a `KeepN` budget counts
  physical states, so SU(2)-graded blocks work with no change here.

Symmetry-agnostic otherwise — a new symmetry reuses it by defining `multiplicity`.
"""
function truncation_plan(
    vals::Dict{K,Vector{Float64}}, trunc::KeepN, sym::AbstractSymmetry
) where {K}
    entries = Tuple{Float64,K,Int,Int}[]                 # (energy, qn, index, weight)
    for (q, ev) in vals
        w = multiplicity(sym, q)
        for (i, e) in enumerate(ev)
            push!(entries, (e, q, i, w))
        end
    end
    sort!(entries; by=first)
    plan = Dict{K,Vector{Int}}()
    cum = 0
    for (k, (e, q, i, w)) in enumerate(entries)
        push!(get!(plan, q, Int[]), i)
        cum += w
        # keep ≥ N physical states, then extend through any (near-)degenerate cluster
        if cum ≥ trunc.N &&
            (k == length(entries) || !isapprox(entries[k + 1][1], e; atol=1e-9))
            break
        end
    end
    for q in keys(plan)
        sort!(plan[q])
    end
    return plan
end
function truncation_plan(
    vals::Dict{K,Vector{Float64}}, trunc::EnergyCut, ::AbstractSymmetry
) where {K}
    plan = Dict{K,Vector{Int}}()
    for (q, ev) in vals
        idx = findall(≤(trunc.Ecut), ev)
        isempty(idx) || (plan[q] = idx)
    end
    return plan
end

# ---- the scheme ------------------------------------------------------------

"""
    bath_sites_in_init(model) -> Int

How many bath (Wilson-chain) orbitals `impurity_init` already incorporates. `0`
for models whose init is the impurity alone (Anderson — `add_site` then attaches
`f₀` with coupling `V₀`); `1` for models whose impurity is exchange-coupled to
`f₀` inside the init (Kondo — the first attach is `f₁`). Lets `nrg_solve` stay
model-generic.

CONTRACT: this must agree with what `impurity_init` actually builds. A model whose
init bakes in `f₀` MUST override this to return ≥1, otherwise `nrg_solve` attaches
`f₀` a second time and silently double-counts a Wilson site (a convergent but wrong
flow). The two are an unenforced pair — keep them in sync when adding a model.
"""
bath_sites_in_init(::AbstractImpurityModel) = 0

"""
    nrg_solve(model::AbstractImpurityModel, alg::NRGAlgorithm) -> NRGResult

Run the generic NRG scheme: discretize the bath, then iteratively enlarge,
diagonalize, rescale by `√Λ`, and truncate.

```
chain = wilson_chain(alg.discretization, model, alg.nsites)
state = impurity_init(model, alg.symmetry, chain)
for n in bath_sites_in_init(model):(alg.nsites - 1)          # n=0: impurity↔f₀ (V₀); n≥1: f↔f (ξ)
    enl  = add_site(state, alg.symmetry; coupling, rescale, onsite)   # √Λ rescale + ξ-hopping
    diag = diagonalize_blocks(enl, alg.symmetry)
    plan = truncation_plan(diag.vals, alg.truncation, alg.symmetry)
    state = update_operators(diag, plan, alg.symmetry)   # subtracts the ground energy
end
```

Wired for `WilsonLog` + `U1U1` + `AndersonModel` (Stage 1). Other symmetries raise
[`EngineUnimplemented`](@ref) until their layer lands (see the roadmap).
"""
function nrg_solve(model::AbstractImpurityModel, alg::NRGAlgorithm)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sym = alg.symmetry
    state = impurity_init(model, sym, chain)        # throws for unwired symmetries
    Λ = alg.discretization.Λ
    sqrtΛ = sqrt(Λ)
    energies = Vector{Vector{Float64}}()
    levels = Vector{Vector{Tuple{Float64,Int}}}()
    scale = Float64[]
    kept = Int[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        enl = add_site(state, sym; coupling, rescale, onsite=chain.onsite[n + 1])
        diag = diagonalize_blocks(enl, sym)
        plan = truncation_plan(diag.vals, alg.truncation, sym)
        state = update_operators(diag, plan, sym)        # truncates + subtracts ground energy
        lv = block_levels(state, sym)                    # (energy, 2Sₙ) pairs, relative to ground
        push!(levels, lv)
        push!(energies, sort!([e for (e, _) in lv]))
        push!(kept, length(lv))
        push!(scale, shell_scale(alg.discretization, n))
    end
    return NRGResult(chain, alg, energies, kept, levels, scale)
end

# The dynamical layer (green_function / spectral / self_energy, Axis 4) is in
# src/spectral.jl and src/self_energy.jl.
