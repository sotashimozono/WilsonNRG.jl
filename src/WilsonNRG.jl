"""
    WilsonNRG

A generic numerical renormalization group (NRG) *scheme* — organised the way
DMRG is: one model-agnostic driver ([`nrg_solve`](@ref)) over orthogonal
*formulation* axes, each a small abstract-type family with one concrete type per
published method.

    Axis 1  AbstractImpurityModel   — the physics            (AndersonModel, …)
    Axis 2  AbstractDiscretization  — Δ(ω) ↦ Wilson chain     (WilsonLog, …)
    Axis 3  AbstractSymmetry        — block / multiplet form  (U1U1, …)
    Axis 4  AbstractSpectralMethod  — eigenflow ↦ A(ω)        (FDM, …)

Implemented now: the U=0 analytic bootstrap and the `WilsonLog` discretization,
each behind a faithfulness gate. The many-body engine lands per the staged
roadmap (see the docs); until then [`nrg_solve`](@ref) raises
[`EngineUnimplemented`](@ref) for the unwired symmetries.

An optional `ITensorMPS` bridge (NRG-as-MPS, Saberi–Weichselbaum–von Delft 2008)
is provided as a package extension and loads only when `ITensorMPS` is present.
"""
module WilsonNRG

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
export shell_scale, thermodynamics, magnetization
export green_function, self_energy, hybridization_function, compare_self_energy
export default_spectral_method, default_self_energy_method
export clebsch_gordan, wigner3j, wigner6j

include("bootstrap.jl")
include("types.jl")
include("discretization.jl")
include("interface.jl")
include("engine_u1u1.jl")
include("thermodynamics.jl")
include("kondo.jl")
include("spectral.jl")
include("cfs.jl")
include("fdm.jl")
include("self_energy.jl")
include("su2.jl")

end # module WilsonNRG
