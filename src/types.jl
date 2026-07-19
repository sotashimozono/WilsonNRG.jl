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
Bulla, Costi & Pruschke, [doi_10.1103_RevModPhys.80.395](@cite).
"""
struct AndersonModel <: AbstractImpurityModel
    U::Float64
    εd::Float64
    Γ::Float64
    D::Float64
end
function AndersonModel(; U::Real, εd::Real=(-U / 2), Γ::Real, D::Real=1.0)
    return AndersonModel(Float64(U), Float64(εd), Float64(Γ), Float64(D))
end

"""
    KondoModel(; J, D = 1.0) <: AbstractImpurityModel

Spin-½ Kondo model: antiferromagnetic exchange `J` to a flat band of half-width
`D`. **Planned** (Stage 5); declared here so the symmetry/spectral machinery can
be exercised against the Kondo fixed point. Ref: Wilson, [doi_10.1103_RevModPhys.47.773](@cite).
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
    CampoOliveira(Λ; z=1.0) <: AbstractDiscretization

`z`-averaging discretization (Campo & Oliveira, [doi_10.1103_PhysRevB.72.104432](@cite)) — shifts the
logarithmic grid by the twist `z ∈ (0,1]` and takes the log-mean representative energy
(Žitko Eq. 32). `wilson_chain` builds the z-shifted chain via Lanczos, so it runs in
[`nrg_solve`](@ref); note this recipe still leaves band-edge artefacts in `band_dos`
(cured by [`ZitkoPruschke`](@ref)).
"""
struct CampoOliveira <: AbstractDiscretization
    Λ::Float64
    z::Float64
end
CampoOliveira(Λ::Real; z::Real=1.0) = CampoOliveira(Float64(Λ), Float64(z))

"""
    ZitkoPruschke(Λ) <: AbstractDiscretization

Improved discretization (Žitko & Pruschke, [doi_10.1103_PhysRevB.79.085106](@cite)): the representative
energies are chosen so the z-averaged band is reproduced *exactly*, `A_{f0}(ω) = ρ(ω)`,
removing the band-edge artefacts of the conventional/Campo–Oliveira schemes (their Eq. 35/36).
The flat-band band-DOS reproduction is available via [`band_dos`](@ref); `wilson_chain`
builds the z-shifted Wilson chain (twist `z`, default `1.0`) via Lanczos, so it runs in
[`nrg_solve`](@ref). z-averaging = averaging observables over several `z ∈ (0,1]`.
"""
struct ZitkoPruschke <: AbstractDiscretization
    Λ::Float64
    z::Float64
end
ZitkoPruschke(Λ::Real; z::Real=1.0) = ZitkoPruschke(Float64(Λ), Float64(z))

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

Supertype of spectral-function formulations (Axis 4a): each builds the impurity
Green's function `G(ω)` via [`green_function`](@ref), from which `A(ω) = -Im G/π`
([`spectral`](@ref)) and the self-energy follow. Diverse methods exist because
getting `A(ω)`/`Σ(ω)` accurately is hard — they are meant to be **compared**
([`compare_self_energy`](@ref)); the robust choice is [`default_spectral_method`](@ref).
"""
abstract type AbstractSpectralMethod end

"`BHP` — Bulla–Hewson–Pruschke T=0 spectral patching (self-energy trick: [doi_10.1088_0953-8984_10_37_021](@cite); broadening/patching per Bulla–Costi–Pruschke [doi_10.1103_RevModPhys.80.395](@cite), §III.B). **Implemented.**"
struct BHP <: AbstractSpectralMethod end
"`DMNRG` — density-matrix NRG spectral function via the off-diagonal reduced DM (Hofstetter, [doi_10.1103_PhysRevLett.85.1508](@cite)). **Implemented.**"
struct DMNRG <: AbstractSpectralMethod end
"`CFS` — complete-Fock-space / TDNRG (Anders & Schiller, [doi_10.1103_PhysRevLett.95.196801](@cite)). **Planned**."
struct CFS <: AbstractSpectralMethod end
"`FDM` — full-density-matrix, sum-rule conserving (Weichselbaum & von Delft, [doi_10.1103_PhysRevLett.99.076402](@cite)). **Planned** (robust default-to-be)."
struct FDM <: AbstractSpectralMethod end

"""
    AbstractSelfEnergyMethod

Supertype of self-energy formulations (Axis 4b): how `Σ(ω)` is extracted from the
Green's function. The choice matters for accuracy — see [`compare_self_energy`](@ref).
"""
abstract type AbstractSelfEnergyMethod end

"""
    SelfEnergyTrick() <: AbstractSelfEnergyMethod

`Σ_σ = U · F_σ / G_σ` with `F_σ = ⟨⟨d_σ n_{-σ}; d†_σ⟩⟩` (Bulla–Hewson–Pruschke,
[doi_10.1088_0953-8984_10_37_021](@cite)). **Robust** — `Σ ∝ U` (exact `0` at `U=0`) and `F/G` shares
poles/broadening so errors largely cancel. The default ([`default_self_energy_method`](@ref)).
"""
struct SelfEnergyTrick <: AbstractSelfEnergyMethod end

"""
    Dyson() <: AbstractSelfEnergyMethod

`Σ_σ = ω - ε_d - Δ(ω) - 1/G_σ(ω)`. Simple, but broadening errors in `G` are
amplified by `1/G`; provided mainly as a comparison baseline.
"""
struct Dyson <: AbstractSelfEnergyMethod end

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
the shape is fixed here so downstream code and the MPS layer (`nrg_mps`) can
depend on it.

- `chain::WilsonChain`       — the bath used
- `algorithm::NRGAlgorithm`  — the scheme configuration
- `energies::Vector{Vector{Float64}}` — rescaled, ground-subtracted eigenvalues kept at each iteration
- `kept::Vector{Int}`        — number of states kept per iteration
- `levels::Vector{Vector{Tuple{Float64,Int}}}` — per iteration, the kept `(energy, spin-label)`
  pairs. **The `Int` is symmetry-dependent**: for abelian symmetries (`U1U1`) it is `2Sₙ` and each
  entry is one physical state; for non-abelian symmetries (`U1SU2`) it is `2S` and each entry is one
  *multiplet* of `2S+1` states — consumers must weight by [`multiplicity`](@ref). (`thermodynamics`/
  `magnetization` therefore currently require `U1U1`; the spectral layer guards likewise.)
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
