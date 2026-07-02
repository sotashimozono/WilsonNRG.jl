"""
    WilsonNRG

A generic numerical renormalization group (NRG) *scheme* — organised the way
DMRG is: one model-agnostic driver ([`nrg_solve`](@ref)) over orthogonal
*formulation* axes, each a small abstract-type family with one concrete type per
published method (each type's citation is its faithfulness gate).

    Axis 1  AbstractImpurityModel   — the physics            AndersonModel, KondoModel
    Axis 2  AbstractDiscretization  — Δ(ω) ↦ Wilson chain     WilsonLog, CampoOliveira, ZitkoPruschke
    Axis 3  AbstractSymmetry        — block / multiplet form  U1U1, U1SU2, SU2SU2
    Axis 4  AbstractSpectralMethod  — eigenflow ↦ A(ω)        BHP, CFS, FDM, DMNRG

On top of the axes: thermodynamics (entropy, χ, Wilson ratio `R=2`) and
magnetization; the self-energy `Σ(ω)` (`self_energy`, trick + Dyson) and the
self-energy-improved Green's function (`improved_green_function`); z-averaged
spectra (Žitko–Pruschke / Oliveira–Oliveira); the NRG-as-MPS structural
reproduction (`nrg_mps`, Saberi 2008); and Kondo-physics validation gates
(`T_K`, universal scaling, Friedel `πΓA(0)=1`).

# Scope & limitations (deliberate — read before trusting a number)
This reproduces the *established single-impurity, single-channel, equilibrium*
NRG genealogy: a benchmark / teaching engine, NOT a production DMFT solver.

  * **Bath** — flat band, constant hybridization `Δ(ω)=Γ` for `|ω|<D`. There is no
    arbitrary `Δ(ω)` / general-DOS Wilson chain, hence no DMFT self-consistency and
    no multi-orbital or multi-channel impurity — the rungs a DFT+DMFT bath solver
    would add next.
  * **Resolution** — single-`z`, moderate `Λ`. The *direct* `A(ω)` is
    broadening-limited (the log-Gaussian washes the `~T_K`-narrow Kondo peak), so
    the Friedel unitary limit `πΓA(0)=1` is recovered via `improved_green_function`
    (`ReΣ(0)=U/2` pins it — the standard accurate route) or by z-averaging, NOT from
    the bare `A(ω)`. Exact pointwise shape wants small `Λ` + many `z`.
  * **Non-abelian finite T** — `FDM` / `DMNRG` are `U1U1`-only (they raise
    `EngineUnimplemented` for other symmetries): the non-abelian thermal reduced
    density matrix needs the QSpace `(2S+1)` multiplet-weight bookkeeping, which is
    not yet machine-precise here. `U1SU2` dynamics are served instead by `CFS`
    (T=0, exact) plus the exact self-energy trick.
  * **SU2SU2 spectra** (the charge-SU(2) Nambu double-tensor) are not implemented.
  * **Method artifacts** — the BHP windowed particle–hole break at `U>0` and the
    single-`z` occupation floor — are catalogued on the docs "Known limitations" page.

The NRG-as-MPS reproduction is pure `LinearAlgebra` (exact for small chains); a
QSpace / `ITensorMPS`-scale bridge is a future extension, not present today.
"""
module WilsonNRG

using CommonSolve: CommonSolve, solve, init, solve!   # the ecosystem solve/init/solve! verbs (re-exported)

# --- U=0 analytic bootstrap (exact reference for the engine) ---
export resonant_level_spectral, friedel_pin, spectral_sum_rule

# --- dispatch axes (the scheme) ---
export AbstractImpurityModel,
    AbstractDiscretization, AbstractSymmetry, AbstractSpectralMethod, AbstractTruncation
export AndersonModel, KondoModel
export WilsonLog, CampoOliveira, ZitkoPruschke
export U1U1, U1SU2, SU2SU2
export AbstractSpectralMethod, BHP, DMNRG, CFS, FDM
export AbstractSelfEnergyMethod, SelfEnergyTrick, Dyson
export KeepN, EnergyCut

# --- core objects + driver ---
export WilsonChain, NRGAlgorithm, NRGResult, EngineUnimplemented
export wilson_chain, asymptotic_hopping, hybridization, bath_coupling, nrg_solve, spectral
export band_dos
export shell_scale, thermodynamics, magnetization, wilson_ratio
export green_function, self_energy, hybridization_function, compare_self_energy
export improved_green_function
export zavg_green_function, zavg_spectral
# --- the reusable impurity-solver seam (DMFT/DMET/benchmark integration point) ---
export solve, init, solve!                              # CommonSolve verbs: solve(problem, solver)
export AbstractImpuritySolver,
    NRGSolver, ImpuritySolution, impurity_solve, spectral_function
export ImpurityProblem, AbstractBath, FlatBand, NumericalBath
export AbstractEigensolver, DenseEigen
export occupation, double_occupancy, quench_dynamics
export default_spectral_method, default_self_energy_method
export nrg_mps, reconstruct_mps, wilson_chain_hamiltonian, best_mps_energy
export clebsch_gordan, wigner3j, wigner6j

include("bootstrap.jl")
include("types.jl")
include("discretization.jl")
include("discretization_zavg.jl")
include("interface.jl")
include("engine_u1u1.jl")
include("thermodynamics.jl")
include("kondo.jl")
include("spectral.jl")
include("cfs.jl")
include("fdm.jl")
include("dmnrg.jl")
include("occupation.jl")
include("tdnrg.jl")
include("self_energy.jl")
include("zavg_spectral.jl")
include("nrg_mps.jl")
include("su2.jl")
include("engine_u1su2.jl")
include("engine_su2su2.jl")
include("cfs_su2.jl")          # CFS spectral for U1SU2 (needs the engine above)
include("spectral_su2.jl")     # self-energy trick (Σ=U·F/G) for U1SU2
include("solver.jl")           # the reusable impurity-solver seam (wraps the above)

end # module WilsonNRG
