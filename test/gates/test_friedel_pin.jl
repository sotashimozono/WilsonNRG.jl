# Faithfulness gate — each @test is a no-cite-without-reproduction check: a cited result holds
# only if `reproduce ≈ reference`. References are grounded against the fetched sources in
# docs/refs and listed in docs/reference.bib (Friedel sum rule: Langreth 1966; spectral sum rule
# + Fermi-liquid pin: Weichselbaum–von Delft 2007, Bulla–Costi–Pruschke 2008).
# Scope: U = 0 bootstrap (analytic, exact). U > 0 (Kondo) is gated on the NRG engine — not yet claimed.

using WilsonNRG, Test

@testset "method-recovery gate · U=0 bootstrap" begin
    Γ = 0.137

    # ---- P1 non-vacuity floor: A(ω) is a genuine peaked Lorentzian, not a stub ----
    @test resonant_level_spectral(0.0; Γ) ≈ 1 / (π * Γ)                       # peak height
    @test resonant_level_spectral(Γ;   Γ) ≈ 1 / (2π * Γ)                      # HWHM ⇒ half peak
    @test resonant_level_spectral(0.0; Γ) > 50 * resonant_level_spectral(50Γ; Γ)  # actually peaked

    # ---- claim `friedel-pin`  [Weichselbaum2007 ∧ BullaCostiPruschke2008] ----
    #   πΓ·A(0) = sin²(π⟨n_σ⟩) = 1 at the symmetric point (⟨n_σ⟩ = 1/2). Bootstrap: U = 0.
    @test friedel_pin(; Γ) ≈ 1.0 rtol = 2e-3
    # universality (the pin is Γ-independent) — guards against a Γ-tuned fluke:
    @test friedel_pin(; Γ = 1.0) ≈ 1.0 rtol = 2e-3
    @test friedel_pin(; Γ = 0.01) ≈ 1.0 rtol = 2e-3

    # ---- claim `spectral-sum-rule`  [BullaCostiPruschke2008: ∫ A dω = 1, exact] ----
    @test spectral_sum_rule(; Γ) ≈ 1.0 rtol = 1e-3
    @test spectral_sum_rule(; Γ = 1.0) ≈ 1.0 rtol = 1e-3
end
