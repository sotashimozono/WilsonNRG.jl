# WilsonNRG.jl

[![docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://codes.sota-shimozono.com/WilsonNRG.jl/stable/)
[![docs: dev](https://img.shields.io/badge/docs-dev-purple.svg)](https://codes.sota-shimozono.com/WilsonNRG.jl/dev/)
[![Julia](https://img.shields.io/badge/julia-v1.10+-9558b2.svg)](https://julialang.org)
[![Code Style: Blue](https://img.shields.io/badge/Code%20Style-Blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

<a id="badge-top"></a>
[![codecov](https://codecov.io/gh/sotashimozono/WilsonNRG.jl/graph/badge.svg?token=Q3oEEiz9A2)](https://codecov.io/gh/sotashimozono/WilsonNRG.jl)
[![Build Status](https://github.com/sotashimozono/WilsonNRG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sotashimozono/WilsonNRG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/main/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **faithful, reproducible implementation of Wilson's Numerical Renormalization
Group (NRG)** for quantum-impurity models — organised as a *generic scheme*, the
way DMRG is a scheme rather than a single model.

WilsonNRG aims to reproduce the published NRG genealogy (Wilson 1975;
Krishna-murthy–Wilkins–Wilson 1980; Anders–Schiller 2005; Weichselbaum–von Delft
2007; Žitko–Pruschke 2009; Bulla–Costi–Pruschke 2008) — *not* to be a novel
solver. Each result is checked against the source it cites.

## The idea: one driver, four dispatch axes

A concrete NRG calculation is one choice per orthogonal *formulation* axis, fed to
a single model-agnostic driver `nrg_solve(model, alg)`:

| Axis | Abstract type | Concrete formulations (= papers) |
|------|---------------|----------------------------------|
| Impurity model | `AbstractImpurityModel` | `AndersonModel`, `KondoModel` |
| Bath discretization | `AbstractDiscretization` | `WilsonLog`, `CampoOliveira`, `ZitkoPruschke` |
| Symmetry | `AbstractSymmetry` | `U1U1`, `U1SU2`, `SU2SU2` |
| Spectral method | `AbstractSpectralMethod` | `BHP`, `DMNRG`, `CFS`, `FDM` |
| Truncation | `AbstractTruncation` | `KeepN`, `EnergyCut` |

Every concrete formulation type names the paper it reproduces, so extending the
method set is *adding a dispatch method*, and a citation becomes a checkable,
dispatchable object rather than prose.

An optional bridge to `ITensorMPS` (NRG-as-MPS; Saberi–Weichselbaum–von Delft
2008) ships as a package extension and loads only when `ITensorMPS` is present.

## Status

This package is built up in reviewable stages, each gated against its source.

| Stage | Scope | State |
|-------|-------|-------|
| 0 | Generic-scheme skeleton; `WilsonLog` discretization; U=0 analytic bootstrap | ✅ implemented + gated |
| 1 | `WilsonLog` + `U1U1` + `AndersonModel` iterative diagonalization (energy flow) | 🚧 |
| 2 | Thermodynamics (entropy, susceptibility) | ⬜ |
| 3 | `FDM` spectral function `A(ω)` | ⬜ |
| 4 | `ITensorMPS` extension (`as_mps`, NRG-vs-VMPS) | ⬜ |
| 5+ | More discretizations / symmetries / models / spectral methods | ⬜ |

Calling `nrg_solve` for a not-yet-wired symmetry raises a clear
`EngineUnimplemented` rather than failing silently.

## Example (available today)

```julia
using WilsonNRG

# U = 0 resonant level — the exact reference the engine must recover
A0 = resonant_level_spectral(0.0; Γ = 0.1)      # Lorentzian peak
friedel_pin(; Γ = 0.1)                          # πΓ·A(0) = 1 (Friedel/Fermi-liquid pin)

# Wilson logarithmic discretization of a flat band
chain = wilson_chain(WilsonLog(2.5), AndersonModel(; U = 0.0, Γ = 0.01), 40)
chain.hopping        # ξₙ → (1 + Λ⁻¹)/2  (KWW 1980 / Bulla 2008)

# The scheme configuration (driver lands in Stage 1)
alg = NRGAlgorithm(; discretization = WilsonLog(2.5), symmetry = U1U1(),
                     truncation = KeepN(1024), nsites = 40)
```

## References

See [`docs/reference.bib`](docs/reference.bib) for the full, DOI-verified NRG
genealogy. License: MIT.
