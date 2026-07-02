# ===========================================================================
#  The reusable impurity-solver seam.
#
#  WilsonNRG is, from the outside, one thing: a map from an impurity PROBLEM (a model
#  coupled to a bath) to its local Green's function / self-energy. This file names that
#  map and fixes a STABLE OUTPUT SHAPE so downstream code — a DMFT/DMET self-consistency
#  loop, or a cross-solver benchmark (e.g. vs a complex-time MPS solver) — depends on the
#  contract, not on the internal `green_function`/`self_energy` calls.
#
#      solve(problem::ImpurityProblem, solver::AbstractImpuritySolver) -> ImpuritySolution
#
#  It is the `CommonSolve.solve` verb (problem first, solver second — the SciML/ecosystem
#  convention), so `init`/`solve!` give a DMFT loop the standard warm-restart shape and a
#  consumer written for the ecosystem drives WilsonNRG with no glue. `AbstractImpuritySolver`
#  is the OPEN contract: an external ED/CT-QMC/complex-time-MPS solver subtypes it + adds one
#  `solve` method. `NRGSolver` is this package's implementation; it DELEGATES to the existing
#  dispatch (a wrapper, not a re-derivation): `G`/`Σ` are exactly `self_energy` +
#  `_green_from_self_energy`, one computation path (DRY), gated in
#  test/gates/test_solver_interface.jl.
#
#  BATH / the DMFT rung. The bath is a first-class part of the problem. `FlatBand` (the
#  model's own flat band) is wired today; `NumericalBath` (an arbitrary self-consistent
#  `Δ(ω)`, what a DMFT loop feeds back) is DECLARED so the contract is bath-general now, but
#  solving it needs the general-DOS Wilson chain (Lanczos tridiagonalization) and currently
#  refuses. Adding it later fills a slot — it does not change this contract.
# ===========================================================================

# ---- the bath ----
"""
    AbstractBath

The bath an impurity couples to, as its local hybridization `Δ(ω)`. Concrete: [`FlatBand`](@ref)
(the model's own flat band, the only one wired today) and [`NumericalBath`](@ref) (an arbitrary
`Δ(ω)` for the DMFT/general-DOS path, declared but not yet implemented).
"""
abstract type AbstractBath end

"""
    FlatBand()

Marker bath: use the impurity model's own flat band `Δ(ω)=Γ` (`|ω|<D`). It carries no data
(Γ, D live on the model), so it can never disagree with the model.
"""
struct FlatBand <: AbstractBath end

"""
    NumericalBath(ω, Δ)

An arbitrary tabulated hybridization `Δ(ω)` — the bath a DMFT self-consistency loop feeds back.
DECLARED so the contract is bath-general now; solving it needs the general-DOS Wilson chain
(Lanczos tridiagonalization) and currently raises [`EngineUnimplemented`](@ref).
"""
struct NumericalBath <: AbstractBath
    ω::Vector{Float64}
    Δ::Vector{ComplexF64}
end

"""
    ImpurityProblem(model, bath=FlatBand())

The impurity problem a solver consumes: an [`AbstractImpurityModel`](@ref) coupled to an
[`AbstractBath`](@ref). The first positional argument of `solve` (`solve(problem,
solver)`, the CommonSolve/SciML convention); a DMFT loop rebuilds the problem with a new
`bath` each iteration and re-`solve`s.
"""
struct ImpurityProblem{M<:AbstractImpurityModel,B<:AbstractBath}
    model::M
    bath::B
end
ImpurityProblem(model::AbstractImpurityModel) = ImpurityProblem(model, FlatBand())

# ---- the eigensolver axis (placeholder; dense is right for NRG's small blocks) ----
"""
    AbstractEigensolver

How each (small, dense, symmetry-blocked) NRG Hamiltonian block is diagonalized. Only
[`DenseEigen`](@ref) is wired — LAPACK `eigen`, the correct default, since NRG keeps ALL low
eigenpairs of small blocks (the regime dense wins and Krylov loses). Declared as a dispatch
axis so a Krylov/GPU/banded backend is an additive `diagonalize_blocks(…, ::YourEigensolver)`
later, without touching this contract.
"""
abstract type AbstractEigensolver end

"""
    DenseEigen()

Dense LAPACK `eigen(Symmetric(·))` — the current (and default) NRG block eigensolver.
"""
struct DenseEigen <: AbstractEigensolver end

# ---- the solver ----
"""
    AbstractImpuritySolver

The OPEN contract: `solve(problem::`[`ImpurityProblem`](@ref)`, solver) -> `[`ImpuritySolution`](@ref).
Another package makes its solver usable by every consumer written against the contract by
subtyping this and adding one `solve(::ImpurityProblem, ::YourSolver)` method.

Behavioural obligations a conforming method MUST honour (this is the contract, beyond the type
signature):
  * **Grid.** Honour a caller-supplied `ω::AbstractVector{<:Real}` and return `G`/`Σ` on
    exactly that grid; with `ω=nothing`, any solver-chosen grid is acceptable.
  * **Real axis, retarded.** `G(ω)` is the RETARDED Green's function on the real axis — a
    natively-Matsubara solver (CT-QMC) must analytically continue before returning.
  * **Self-energy.** `Σ` uses the Dyson convention `Σ = ω − εd − Δ − G⁻¹`; return `nothing`
    if the solver does not produce a trustworthy `Σ`.

Dispatch on your SPECIFIC supported model/bath types (not `AbstractImpurityModel` generically)
and let the fallback throw [`EngineUnimplemented`](@ref) for the rest — the package's
precondition-as-dispatch idiom. [`NRGSolver`](@ref) is this package's implementation.
"""
abstract type AbstractImpuritySolver end

"""
    NRGSolver(; algorithm, spectral_method, self_energy_method, eigensolver)

WilsonNRG as an [`AbstractImpuritySolver`](@ref). `algorithm` ([`NRGAlgorithm`](@ref))
configures the ITERATIVE ENGINE that produces the eigenflow (discretization/symmetry/
truncation/nsites); `spectral_method` + `self_energy_method` configure how observables are read
OFF that eigenflow — orthogonal concerns (thermodynamics needs only the former), which is why
they are separate structs. `eigensolver` is the (currently dense-only) diagonalization backend.
All fields default, so `NRGSolver()` runs; override one axis via
`NRGSolver(; algorithm=NRGAlgorithm(; discretization=WilsonLog(3.0)))`.
"""
Base.@kwdef struct NRGSolver{
    A<:NRGAlgorithm,
    S<:AbstractSpectralMethod,
    E<:AbstractSelfEnergyMethod,
    X<:AbstractEigensolver,
} <: AbstractImpuritySolver
    algorithm::A = NRGAlgorithm(; discretization=WilsonLog(2.0))
    spectral_method::S = default_spectral_method()
    self_energy_method::E = default_self_energy_method()
    eigensolver::X = DenseEigen()
end

# ---- the output ----
"""
    ImpuritySolution(ω, G, Σ, n, solver)

The stable output of `solve` — arrays over a common real-frequency grid `ω`, plus
provenance. Only `G` is universally present; `Σ`/`n` are `nothing` when a solver does not
produce them (so an external `G`-only solver fits the contract without fabricating a `Σ`):

  * `ω::Vector{Float64}`                   — real frequencies
  * `G::Vector{ComplexF64}`                — retarded impurity Green's function `G(ω)`
  * `Σ::Union{Vector{ComplexF64},Nothing}` — self-energy `Σ(ω)` (the DMFT currency), or `nothing`
  * `n::Union{Float64,Nothing}`            — impurity occupation `⟨n_d⟩` (Friedel/Luttinger), or `nothing`
  * `solver::AbstractImpuritySolver`       — the solver that produced this (provenance)

The spectral function is DERIVED, never stored: [`spectral_function`](@ref)`(sol) = -Im G/π`
(so it can never disagree with `G`). NOT parametrized on the solver type, so a heterogeneous
`Vector{ImpuritySolution}` (a DMFT sweep / cross-solver benchmark) stays concretely typed.
"""
struct ImpuritySolution
    ω::Vector{Float64}
    G::Vector{ComplexF64}
    Σ::Union{Vector{ComplexF64},Nothing}
    n::Union{Float64,Nothing}
    solver::AbstractImpuritySolver
end

"""
    spectral_function(sol::ImpuritySolution) -> Vector{Float64}

The spectral function `A(ω) = -Im G(ω)/π`, derived from `sol.G` on demand (not stored).
"""
spectral_function(sol::ImpuritySolution) = @. -imag(sol.G) / π

function Base.show(io::IO, sol::ImpuritySolution)
    Σtag = sol.Σ === nothing ? "" : ", Σ"
    ntag = sol.n === nothing ? "" : ", n=$(round(sol.n; digits=4))"
    return print(
        io, "ImpuritySolution($(length(sol.ω)) freqs; G$Σtag$ntag | $(sol.solver))"
    )
end

# ---- solve (CommonSolve; problem-first) ----
"""
    solve(problem::ImpurityProblem, solver::AbstractImpuritySolver; ω=nothing, with_occupation=true, kw...) -> ImpuritySolution

Solve `problem` with `solver` — the `CommonSolve.solve` verb (with `init`/`solve!` for a
warm-restart DMFT loop). For [`NRGSolver`](@ref) on a flat-band [`AndersonModel`](@ref) it
delegates to [`self_energy`](@ref) (one NRG sweep) + the shared `_green_from_self_energy`, so
`G`/`Σ` equal the direct-call composition byte-for-byte (DRY). `ω` and any broadening kwargs
(`b`, `window`) pass through; `with_occupation` toggles the extra `⟨n_d⟩` sweep (computed only
where `occupation` is supported, i.e. `U1U1`). Unsupported `(model, bath, solver)` triples
raise [`EngineUnimplemented`](@ref).
"""
function CommonSolve.solve(
    prob::ImpurityProblem{<:AndersonModel,FlatBand},
    s::NRGSolver;
    ω=nothing,
    with_occupation::Bool=true,
    kw...,
)
    model = prob.model
    se = self_energy(
        s.spectral_method, model, s.algorithm; via=s.self_energy_method, ω, kw...
    )
    G = _green_from_self_energy(model, se.ω, se.Σ)
    n = if (with_occupation && s.algorithm.symmetry isa U1U1)
        occupation(model, s.algorithm).total
    else
        nothing
    end
    return ImpuritySolution(
        collect(float.(se.ω)), collect(ComplexF64, G), collect(ComplexF64, se.Σ), n, s
    )
end
# open-contract fallback — an unsupported (model, bath, solver) triple refuses cleanly (not MethodError)
function CommonSolve.solve(prob::ImpurityProblem, s::AbstractImpuritySolver; kw...)
    return throw(
        EngineUnimplemented(
            "solve is not implemented for ($(typeof(prob.model)), $(typeof(prob.bath)), " *
            "$(typeof(s))); (AndersonModel, FlatBand, NRGSolver) is available.",
        ),
    )
end

# init / solve! — the CommonSolve warm-restart shape. NRG has no nontrivial cache (a new bath ⇒
# a new Wilson chain ⇒ a fresh sweep), so the cache just bundles the call; it exists so a DMFT
# loop can use the ecosystem-standard init→solve! uniformly across solvers that DO warm-restart.
struct ImpuritySolveCache{P<:ImpurityProblem,S<:AbstractImpuritySolver,K}
    problem::P
    solver::S
    kwargs::K
end
function CommonSolve.init(prob::ImpurityProblem, s::AbstractImpuritySolver; kw...)
    return ImpuritySolveCache(prob, s, kw)
end
CommonSolve.solve!(c::ImpuritySolveCache) = solve(c.problem, c.solver; c.kwargs...)

# ---- convenience aliases onto the same seam (descriptive entry points) ----
"""
    impurity_solve([solver=NRGSolver()], model; kw...) -> ImpuritySolution

Convenience wrapper for the flat-band problem: `solve(ImpurityProblem(model), solver; kw...)`.
"""
function impurity_solve(model::AbstractImpurityModel; kw...)
    return solve(ImpurityProblem(model), NRGSolver(); kw...)
end
function impurity_solve(s::AbstractImpuritySolver, model::AbstractImpurityModel; kw...)
    return solve(ImpurityProblem(model), s; kw...)
end
