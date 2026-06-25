module WilsonNRG

export resonant_level_spectral, friedel_pin, spectral_sum_rule

# ── U = 0 bootstrap: the non-interacting symmetric Anderson impurity (resonant level) ──
# At ε_d = 0 (particle–hole symmetric) and hybridization width Γ, the impurity spectral
# function is exactly a Lorentzian:
#
#     A(ω) = (Γ/π) / (ω² + Γ²)
#
# This is the analytic reference the NRG engine must recover as U → 0, and the bootstrap for
# the faithfulness gates (Friedel pin, spectral sum rule) before U > 0 (the Kondo regime,
# which needs the full NRG) is handled.

"""
    resonant_level_spectral(ω; Γ)

Exact impurity spectral function `A(ω)` of the U=0 symmetric Anderson model (resonant level),
hybridization width `Γ`. Lorentzian of half-width `Γ`.
"""
resonant_level_spectral(ω; Γ::Real) = (Γ / π) / (ω^2 + Γ^2)

"""
    friedel_pin(; Γ)

`πΓ·A(0)` for the U=0 symmetric resonant level. Equals the general Friedel/Fermi-liquid
pinning value `sin²(π⟨n_σ⟩)` evaluated at half-filling (`⟨n_σ⟩ = 1/2 ⇒ 1`).

Grounded references (docs/refs): Weichselbaum & von Delft, PRL 99, 076402 (2007) —
"πΓ A^exact_{T=0} = sin²(π⟨c†_0σ c_0σ⟩_0)"; Bulla, Costi & Pruschke, RMP 80, 395 (2008).
"""
friedel_pin(; Γ::Real) = π * Γ * resonant_level_spectral(0.0; Γ)

"""
    spectral_sum_rule(; Γ, W = 1e4Γ, n = 2_000_001)

Numerically integrate `∫ A(ω) dω` (trapezoid over `[-W, W]`) for the U=0 resonant level.
Exact value is 1. Grounded reference (docs/refs): Bulla, Costi & Pruschke, RMP 80, 395 (2008)
— "∫ dω A_σ(ω,T) = 1 is, by construction, fulfilled exactly".
"""
function spectral_sum_rule(; Γ::Real, W::Real = 1.0e4 * Γ, n::Int = 2_000_001)
    ωs = range(-W, W; length = n)
    A = resonant_level_spectral.(ωs; Γ)
    return (sum(A) - (A[1] + A[end]) / 2) * step(ωs)   # trapezoid
end

end # module WilsonNRG
