# ===========================================================================
#  U = 0 bootstrap: the non-interacting symmetric Anderson impurity
#  (resonant level). The analytic reference the NRG engine must recover as
#  U → 0, and the seed of the faithfulness gates (Friedel pin, spectral sum
#  rule) before the U > 0 / Kondo regime — which needs the full engine — is run.
#
#  At εd = 0 (particle–hole symmetric) and hybridization width Γ, the impurity
#  spectral function is exactly a Lorentzian:
#
#      A(ω) = (Γ/π) / (ω² + Γ²)
# ===========================================================================

"""
    resonant_level_spectral(ω; Γ)

Exact impurity spectral function `A(ω)` of the U=0 symmetric Anderson model
(resonant level), hybridization width `Γ`. Lorentzian of half-width `Γ`.
"""
resonant_level_spectral(ω; Γ::Real) = (Γ / π) / (ω^2 + Γ^2)

"""
    friedel_pin(; Γ)

`πΓ·A(0)` for the U=0 symmetric resonant level. Equals the general
Friedel/Fermi-liquid pinning value `sin²(π⟨n_σ⟩)` at half-filling
(`⟨n_σ⟩ = 1/2 ⇒ 1`).

Grounded references (docs/refs): Langreth, [doi_10.1103_PhysRev.150.516](@cite) — primary;
Weichselbaum & von Delft, [doi_10.1103_PhysRevLett.99.076402](@cite) — "πΓ A^exact_{T=0} =
sin²(π⟨c†_0σ c_0σ⟩₀)"; Bulla, Costi & Pruschke, [doi_10.1103_RevModPhys.80.395](@cite).
"""
friedel_pin(; Γ::Real) = π * Γ * resonant_level_spectral(0.0; Γ)

"""
    spectral_sum_rule(; Γ, W = 1e4Γ, n = 2_000_001)

Numerically integrate `∫ A(ω) dω` (trapezoid over `[-W, W]`) for the U=0
resonant level. Exact value is 1. Grounded reference (docs/refs): Bulla, Costi &
Pruschke, [doi_10.1103_RevModPhys.80.395](@cite) — "∫ dω A_σ(ω,T) = 1 is, by construction, fulfilled
exactly".
"""
function spectral_sum_rule(; Γ::Real, W::Real=1.0e4 * Γ, n::Int=2_000_001)
    ωs = range(-W, W; length=n)
    A = resonant_level_spectral.(ωs; Γ)
    return (sum(A) - (A[1] + A[end]) / 2) * step(ωs)   # trapezoid
end
