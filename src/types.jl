# ===========================================================================
#  Dispatch axes of the generic NRG scheme.
#
#  WilsonNRG is organised the way DMRG is: one model-agnostic driver
#  (`nrg_solve`) over a handful of orthogonal *formulation* axes, each a small
#  abstract-type family. A concrete calculation is the choice of one type per
#  axis. Following the project's "論文ごと・formulation ごとに区別" principle,
#  every concrete formulation type names the paper it reproduces, so a citation
#  is a dispatchable, gate-able object rather than prose.
#
#  Axis 1  AbstractImpurityModel    — the physics (the "Hamiltonian you hand DMRG")
#  Axis 2  AbstractDiscretization   — Δ(ω) ↦ Wilson chain {εₙ, ξₙ}      (per paper)
#  Axis 3  AbstractSymmetry         — conserved-charge / multiplet structure
#  Axis 4  AbstractSpectralMethod   — eigenflow ↦ observables           (per paper)
#  (+ AbstractTruncation            — the per-iteration keep policy)
# ===========================================================================

# ---------------------------------------------------------------------------
# Axis 1 — impurity model
# ---------------------------------------------------------------------------

"""
    AbstractImpurityModel

Supertype of quantum-impurity models. A model supplies its impurity Hilbert
space + Hamiltonian, its coupling to the conduction bath, and the hybridization
function `Δ(ω)` the [`wilson_chain`](@ref) is built from. Parallels
`ITensorModels.AbstractLatticeModel`: concrete subtypes override the engine seam
([`impurity_init`](@ref) etc.) rather than living in a registry.
"""
abstract type AbstractImpurityModel end

"""
    AndersonModel(; U, εd = -U/2, Γ, D = 1.0) <: AbstractImpurityModel

Single-impurity Anderson model: interaction `U`, impurity level `εd`
(default `-U/2` = particle–hole-symmetric point), hybridization width `Γ`,
flat conduction band of half-width `D`. The `U = 0` limit is the exact
resonant level reproduced by [`resonant_level_spectral`](@ref).

Refs: Krishna-murthy, Wilkins & Wilson, PRB 21, 1003/1044 (1980);
Bulla, Costi & Pruschke, RMP 80, 395 (2008).
"""
struct AndersonModel <: AbstractImpurityModel
    U::Float64
    εd::Float64
    Γ::Float64
    D::Float64
end
function AndersonModel(; U::Real, εd::Real=-U / 2, Γ::Real, D::Real=1.0)
    AndersonModel(Float64(U), Float64(εd), Float64(Γ), Float64(D))
end

"""
    KondoModel(; J, D = 1.0) <: AbstractImpurityModel

Spin-½ Kondo model: antiferromagnetic exchange `J` to a flat band of half-width
`D`. **Planned** (Stage 5); declared here so the symmetry/spectral machinery can
be exercised against the Kondo fixed point. Ref: Wilson, RMP 47, 773 (1975).
"""
struct KondoModel <: AbstractImpurityModel
    J::Float64
    D::Float64
end
KondoModel(; J::Real, D::Real=1.0) = KondoModel(Float64(J), Float64(D))

# ---------------------------------------------------------------------------
# Axis 2 — bath discretization  (Δ(ω) ↦ Wilson chain)
# ---------------------------------------------------------------------------

"""
    AbstractDiscretization

Supertype of bath-discretization formulations: the map from a hybridization
function `Δ(ω)` to Wilson-chain coefficients `{εₙ, ξₙ}` via [`wilson_chain`](@ref).
"""
abstract type AbstractDiscretization end

"""
    WilsonLog(Λ) <: AbstractDiscretization

Wilson's logarithmic discretization, `Λ > 1`. For a particle–hole-symmetric flat
band this gives the closed-form hoppings `ξₙ` (KWW 1980 Eq. 2.15 / Bulla 2008
Eq. 32). **Implemented.**
"""
struct WilsonLog <: AbstractDiscretization
    Λ::Float64
    function WilsonLog(Λ::Real)
        Λ > 1 || throw(ArgumentError("WilsonLog: Λ must be > 1 (got $Λ)"))
        return new(Float64(Λ))
    end
end

"""
    CampoOliveira(Λ, z) <: AbstractDiscretization

`z`-averaging discretization (Campo & Oliveira, PRB 72, 104432 (2005)) — shifts
the logarithmic grid by `z ∈ (0,1]` to suppress discretization artifacts.
**Planned** (Stage 5).
"""
struct CampoOliveira <: AbstractDiscretization
    Λ::Float64
    z::Float64
end

"""
    ZitkoPruschke(Λ) <: AbstractDiscretization

Adaptive discretization for arbitrary `Δ(ω)` (Žitko & Pruschke, PRB 79, 085106
(2009)). **Planned** (Stage 5).
"""
struct ZitkoPruschke <: AbstractDiscretization
    Λ::Float64
end

# ---------------------------------------------------------------------------
# Axis 3 — symmetry  (conserved charges → multiplet / block structure)
# ---------------------------------------------------------------------------

"""
    AbstractSymmetry

Supertype of symmetry settings. The symmetry fixes the local multiplet basis,
the fermionic operators (signs / Clebsch–Gordan recoupling) and the block
structure of the iterated Hamiltonian — the axis the core driver dispatches on,
analogous to QN-graded indices in DMRG.
"""
abstract type AbstractSymmetry end

"""
    U1U1() <: AbstractSymmetry

U(1) charge ⊗ U(1) spin-z `(Q, Sₙ)` — the abelian setting of
`siteinds("Electron"; conserve_qns = true)` and of the `NonHermitianNRG`
reference. **Target of Stage 1.**
"""
struct U1U1 <: AbstractSymmetry end

"""
    U1SU2() <: AbstractSymmetry

U(1) charge ⊗ SU(2) spin `(Q, S)` — total-spin multiplets, needs Clebsch–Gordan
recoupling. **Planned** (Stage 6).
"""
struct U1SU2 <: AbstractSymmetry end

"""
    SU2SU2() <: AbstractSymmetry

SU(2) charge (Nambu isospin) ⊗ SU(2) spin — the maximal symmetry of the
symmetric Anderson model (Jayaprakash; Anders & Schiller). **Planned** (Stage 6).
"""
struct SU2SU2 <: AbstractSymmetry end

# ---------------------------------------------------------------------------
# Axis 4 — spectral method  (eigenflow → A(ω))
# ---------------------------------------------------------------------------

"""
    AbstractSpectralMethod

Supertype of spectral-function formulations consuming an [`NRGResult`](@ref) via
[`spectral`](@ref).
"""
abstract type AbstractSpectralMethod end

"`BHP` — Bulla–Hewson–Pruschke spectral patching (Bulla et al. 1998). Planned (Stage 4)."
struct BHP <: AbstractSpectralMethod end
"`DMNRG` — density-matrix NRG (Hofstetter, PRL 85, 1508 (2000)). Planned (Stage 4)."
struct DMNRG <: AbstractSpectralMethod end
"`CFS` — complete-Fock-space / TDNRG (Anders & Schiller, PRL 95, 196801 (2005)). Planned (Stage 4)."
struct CFS <: AbstractSpectralMethod end
"`FDM` — full-density-matrix, sum-rule conserving (Weichselbaum & von Delft, PRL 99, 076402 (2007)). Target of Stage 3."
struct FDM <: AbstractSpectralMethod end

# ---------------------------------------------------------------------------
# Truncation policy
# ---------------------------------------------------------------------------

"""
    AbstractTruncation

Per-iteration state-keeping policy consumed by [`truncation_plan`](@ref).
"""
abstract type AbstractTruncation end

"`KeepN(N)` — keep the lowest-energy `N` states/multiplets each iteration."
struct KeepN <: AbstractTruncation
    N::Int
end

"`EnergyCut(Ecut)` — keep all states below rescaled energy `Ecut` each iteration."
struct EnergyCut <: AbstractTruncation
    Ecut::Float64
end

# ---------------------------------------------------------------------------
# Core objects: chain, algorithm, result
# ---------------------------------------------------------------------------

"""
    WilsonChain

Output of [`wilson_chain`](@ref): the discretized bath as a tight-binding chain.

- `onsite::Vector{Float64}`  — `εₙ` (all zero for a symmetric flat band)
- `hopping::Vector{Float64}` — `ξₙ`, the **dimensionless** hoppings; the physical
  `tₙ = Λ^{-n/2} ξₙ` scale is supplied by the per-iteration `√Λ` rescaling in the
  driver (the `NonHermitianNRG`/Bulla convention). Declared here, not folded in.
- `disc::AbstractDiscretization` — the formulation that produced it.
"""
struct WilsonChain
    onsite::Vector{Float64}
    hopping::Vector{Float64}
    disc::AbstractDiscretization
end

Base.length(c::WilsonChain) = length(c.hopping)

"""
    NRGAlgorithm(; discretization, symmetry = U1U1(), truncation = KeepN(1024), nsites = 40)

The "scheme configuration": one choice per dispatch axis plus the chain length
`nsites`. The discretization carries `Λ`, which is also the per-iteration energy
rescaling, so it is not duplicated here.
"""
struct NRGAlgorithm{D<:AbstractDiscretization,S<:AbstractSymmetry,T<:AbstractTruncation}
    discretization::D
    symmetry::S
    truncation::T
    nsites::Int
end
function NRGAlgorithm(;
    discretization::AbstractDiscretization,
    symmetry::AbstractSymmetry=U1U1(),
    truncation::AbstractTruncation=KeepN(1024),
    nsites::Integer=40,
)
    return NRGAlgorithm(discretization, symmetry, truncation, Int(nsites))
end

"""
    NRGResult

Result of an [`nrg_solve`](@ref) run: the energy flow plus the data needed by the
spectral/MPS layers. Fields are populated by the iterative engine (Stage 1+);
the shape is fixed here so downstream code and the `ext/` MPS bridge can depend
on it.

- `chain::WilsonChain`       — the bath used
- `algorithm::NRGAlgorithm`  — the scheme configuration
- `energies::Vector{Vector{Float64}}` — rescaled, ground-subtracted eigenvalues kept at each iteration
- `kept::Vector{Int}`        — number of states kept per iteration
- `levels::Vector{Vector{Tuple{Float64,Int}}}` — per iteration, the kept `(energy, 2Sₙ)` pairs
  (the quantum-number-resolved spectrum the thermodynamics/spectral layers consume)
- `scale::Vector{Float64}`   — `ωₙ`, the characteristic energy (shell) scale of each iteration
"""
struct NRGResult
    chain::WilsonChain
    algorithm::NRGAlgorithm
    energies::Vector{Vector{Float64}}
    kept::Vector{Int}
    levels::Vector{Vector{Tuple{Float64,Int}}}
    scale::Vector{Float64}
end
