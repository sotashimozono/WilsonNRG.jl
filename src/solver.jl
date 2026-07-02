# ===========================================================================
#  The reusable impurity-solver seam.
#
#  WilsonNRG is, from the outside, one thing: a map from an impurity problem to its
#  local Green's function / self-energy. This file gives that map a NAME and a STABLE
#  OUTPUT SHAPE so downstream code — a DMFT/DMET self-consistency loop, or a
#  cross-solver benchmark (e.g. vs a complex-time MPS solver) — can depend on the
#  contract instead of on the internal `green_function` / `self_energy` calls.
#
#      impurity_solve(solver, model) -> ImpuritySolution(ω, G, Σ, A)
#
#  `AbstractImpuritySolver` is the OPEN contract: another package can subtype it and
#  add its own `impurity_solve` method, and a DMFT loop written against the contract
#  will drive either solver unchanged. `NRGSolver` is this package's implementation —
#  a thin bundle of the four dispatch-axis choices (as an `NRGAlgorithm`) plus the
#  spectral method and self-energy route, each defaulted. The solver DELEGATES to the
#  existing dispatch (it is a wrapper, not a re-derivation): its `G`/`Σ` are exactly
#  `self_energy` + [`_green_from_self_energy`](@ref), so there is one computation path
#  (DRY), gated in test/gates/test_solver_interface.jl.
#
#  SCOPE / the DMFT rung. The bath is currently the model's flat band `Δ(ω)=Γ`. A DMFT
#  loop needs to INJECT a general, self-consistent `Δ(ω)` — that (general-DOS Wilson
#  chain via Lanczos tridiagonalization) is the tracked next step; the seam here is
#  shaped for it, but `impurity_solve` does not yet accept an arbitrary `Δ(ω)`.
# ===========================================================================

"""
    AbstractImpuritySolver

The open contract for "given an impurity model (and, in future, a bath `Δ(ω)`),
return its local Green's function and self-energy". Implement
[`impurity_solve`](@ref)`(::YourSolver, model) -> `[`ImpuritySolution`](@ref) to make a
new solver (NRG here; an external ED / CT-QMC / complex-time-MPS solver elsewhere)
usable by any consumer written against the contract — the point of the abstraction.
[`NRGSolver`](@ref) is this package's implementation.
"""
abstract type AbstractImpuritySolver end

"""
    NRGSolver(; algorithm, spectral_method, self_energy_method)

WilsonNRG as an [`AbstractImpuritySolver`](@ref): a bundle of the scheme configuration
([`NRGAlgorithm`](@ref) — the four dispatch axes + chain length) with the spectral
method that builds `G` and the self-energy route. All fields default, so `NRGSolver()`
is a runnable solver (`WilsonLog(Λ=2)`, `U1U1`, `KeepN(1024)`, `nsites=40`, `BHP`,
[`SelfEnergyTrick`](@ref)); override any one axis, e.g.
`NRGSolver(; algorithm=NRGAlgorithm(; discretization=WilsonLog(3.0), truncation=KeepN(500)))`.
"""
Base.@kwdef struct NRGSolver{
    A<:NRGAlgorithm,S<:AbstractSpectralMethod,E<:AbstractSelfEnergyMethod
} <: AbstractImpuritySolver
    algorithm::A = NRGAlgorithm(; discretization=WilsonLog(2.0))
    spectral_method::S = default_spectral_method()
    self_energy_method::E = default_self_energy_method()
end

"""
    ImpuritySolution(ω, G, Σ, A, solver)

The stable output of [`impurity_solve`](@ref) — parallel arrays over a common real-
frequency grid `ω`, plus the `solver` that produced them (provenance):

  * `ω::Vector{Float64}`      — real frequencies
  * `G::Vector{ComplexF64}`   — retarded impurity Green's function `G(ω)`
  * `Σ::Vector{ComplexF64}`   — impurity self-energy `Σ(ω)` (the DMFT currency)
  * `A::Vector{Float64}`      — spectral function `A(ω) = -Im G(ω)/π`

`G` is the self-energy-improved Green's function (`1/(ω-εd-Δ-Σ)`) — the accurate NRG
route (see [`improved_green_function`](@ref)), not the broadening-limited direct `A(ω)`.
"""
struct ImpuritySolution{S<:AbstractImpuritySolver}
    ω::Vector{Float64}
    G::Vector{ComplexF64}
    Σ::Vector{ComplexF64}
    A::Vector{Float64}
    solver::S
end

function Base.show(io::IO, sol::ImpuritySolution)
    return print(
        io,
        "ImpuritySolution(",
        length(sol.ω),
        " frequencies: ω, G, Σ, A | ",
        sol.solver,
        ")",
    )
end

"""
    impurity_solve([solver=NRGSolver()], model; ω=nothing, kw...) -> ImpuritySolution

Solve the impurity `model` with `solver` and return an [`ImpuritySolution`](@ref)
`(ω, G, Σ, A)`. This is the reusable seam: a DMFT/DMET loop, or a benchmark comparing
solvers, calls `impurity_solve` and reads `.G`/`.Σ`, never the internal engine.

For [`NRGSolver`](@ref) it delegates to [`self_energy`](@ref) (one NRG sweep) and
reconstructs `G` from that `Σ` via the shared internal `_green_from_self_energy`, so the
result is byte-for-byte the composition of the existing calls (DRY). `ω` and any
broadening kwargs (`b`, `window`, …) pass through. `AndersonModel` only for now (the
self-energy defines `Σ`); other models raise [`EngineUnimplemented`](@ref).
"""
function impurity_solve(s::NRGSolver, model::AndersonModel; ω=nothing, kw...)
    se = self_energy(
        s.spectral_method, model, s.algorithm; via=s.self_energy_method, ω, kw...
    )
    G = _green_from_self_energy(model, se.ω, se.Σ)
    A = @. -imag(G) / π
    return ImpuritySolution(
        collect(float.(se.ω)), collect(ComplexF64, G), collect(ComplexF64, se.Σ), A, s
    )
end
function impurity_solve(model::AbstractImpurityModel; kw...)
    impurity_solve(NRGSolver(), model; kw...)
end
# open-contract fallback: a solver/model pair with no self-energy route refuses cleanly
function impurity_solve(::AbstractImpuritySolver, model::AbstractImpurityModel; kw...)
    return throw(
        EngineUnimplemented(
            "impurity_solve on $(typeof(model)) is not implemented; AndersonModel is available.",
        ),
    )
end
