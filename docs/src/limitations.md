# Known limitations & method accuracy

WilsonNRG.jl reproduces the published NRG genealogy *faithfully* — which also means it
reports, rather than hides, where a method's own approximation shows up. Every
faithfulness gate under `test/gates/` checks a claim against an **independent**
closed-form / cross-method / structural target, so a deviation that survives is a real
property of the method, not a test artifact. When a gate surfaces such a deviation the
policy is to **probe it** (does it shrink with more kept states / a finer grid, or is it
a floor?) and then **record it honestly** — as a `@test_broken`, or a method-honest
tolerance with the reasoning in-file — never to loosen a tolerance until the test passes.

This page collects the deviations currently recorded that way, with their evidence and
the route to tightening them.

## BHP spectral function is not particle–hole symmetric at U > 0

At the symmetric point εd = −U/2 the impurity spectral function must satisfy
A(ω) = A(−ω). The complete-basis methods (`CFS`, `FDM`, `DMNRG`) meet this to below
0.3 % of max A at every U. `BHP` (Bulla–Hewson–Pruschke windowed patching) meets it at
U = 0 but breaks it once U > 0:

| U | max \|A(ω) − A(−ω)\| (BHP) | CFS / FDM / DMNRG |
|:---|:---|:---|
| 0.0 | symmetric (single resonant level) | ✓ |
| 0.3 | ≈ 1.8  (~30 % of max A) | ✓ |
| 0.6 | ≈ 2.2  (~40 % of max A) | ✓ |

**Cause.** BHP windows each shell's rescaled excitations into `[w, w√Λ]` and patches
across shells on the logarithmic ω grid; the +ω and −ω branches are windowed and
truncated independently, so the many-pole Kondo + Hubbard-satellite spectrum at U > 0 is
not reproduced symmetrically. The same windowing makes the BHP sum rule undersum at
large U (∫A ≈ 0.69 at U = 0.6). Production NRG codes routinely post-symmetrize A at the
symmetric point for exactly this reason.

**Status.** Recorded as `@test_broken` in `test_spectral_sumrules.jl` (not a loosened
tolerance); the tight ∫A = 1 and p-h targets are asserted on the complete-basis methods,
and BHP keeps only a loose sanity floor on its sum rule. Prefer `CFS` / `FDM` / `DMNRG`
for U > 0 spectra. The natural fix is an opt-in post-symmetrization
A(ω) → ½·[A(ω) + A(−ω)] at the symmetric point.
Tracking: [issue #33](https://github.com/sotashimozono/WilsonNRG.jl/issues/33).

**References.** Bulla, Hewson & Pruschke, *J. Phys.: Condens. Matter* **10**, 8365
(1998); Bulla, Costi & Pruschke, *Rev. Mod. Phys.* **80**, 395 (2008).

## Static occupation carries a ~1 % single-z accuracy floor at U > 0

At εd = −U/2 particle–hole symmetry pins ⟨n_d⟩ = 1 exactly. The `occupation` estimator
(a removal/hole-weight sum over the complete Fock space) instead returns
⟨n_d⟩ ≈ 1.009–1.016 for U ∈ [0.4, 0.8] at Λ = 2.5, single-z — a ~1 % particle–hole
*charge* asymmetry. The **spin** pin ⟨n_{d↑}⟩ = ⟨n_{d↓}⟩ holds to machine precision, so
this is an accuracy floor, not a symmetry bug in the operator handling.

The deviation does **not** vanish with more kept states, nor with a finer grid at fixed
truncation:

| control (U = 0.6) | \|⟨n_d⟩ − 1\| |
|:---|:---|
| KeepN 256 / 400 / 600 / 900 (nsites scaled) | 0.0134 / 0.0120 / 0.0116 / 0.0157 |
| Λ = 1.7 / 2.0 / 2.5 / 3.0 (KeepN 500) | 0.043 / 0.019 / 0.013 / 0.012 |

It is the same ~1 % budget as the CFS spectral sum-rule incompleteness (∫A ∈ [0.97, 1.05])
— exact at U = 0, growing with U, consistent with a discretization/truncation floor.

**Status.** `test_ph_symmetry.jl` asserts ⟨n_d⟩ = 1 tightly only where it is exact
(U = 0) and to honest single-z NRG accuracy (2 %) for U > 0; the tightly-preserved p-h
content is carried instead by the exact spin pin, the Fermi-liquid self-energy pin
ReΣ(0) = U/2, and the many-body spectrum blocks E[(Q,D)] = E[(2N−Q,D)]. Reaching sub-%
accuracy needs z-averaging of `occupation` (and/or a Λ → 1 extrapolation), the standard
route for high-precision static NRG properties.
Tracking: [issue #34](https://github.com/sotashimozono/WilsonNRG.jl/issues/34).

**References.** Krishna-murthy, Wilkins & Wilson, *Phys. Rev. B* **21**, 1044 (1980);
Žitko & Pruschke, *Phys. Rev. B* **79**, 085106 (2009).
