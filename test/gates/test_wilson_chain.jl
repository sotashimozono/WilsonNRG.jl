# Faithfulness gate — Wilson logarithmic discretization (Axis 2, formulation `WilsonLog`).
# Each @test is a no-cite-without-reproduction check against the closed-form Wilson chain:
#   ξₙ = (1+Λ⁻¹)/2 · (1−Λ^{−n−1}) / √[(1−Λ^{−2n−1})(1−Λ^{−2n−3})]   (n = 0,1,…)
# grounded in KWW 1980 (PRB 21, 1003), Eq. 2.15, and Bulla–Costi–Pruschke 2008
# (RMP 80, 395), Eq. 32. Exact (closed form), so this is a tier-1 constraint, no rtol slack.
# Scope: particle–hole-symmetric flat band (εₙ = 0). Non-flat baths gate on Stage-5 adaptive
# discretization — not yet claimed.

using WilsonNRG, Test
using WilsonNRG: asymptotic_hopping

@testset "method-recovery gate · WilsonLog discretization" begin
    model = AndersonModel(; U=0.0, Γ=0.01, D=1.0)   # band only enters via D; ξₙ is universal

    for Λ in (1.5, 2.0, 2.5, 3.0)
        chain = wilson_chain(WilsonLog(Λ), model, 60)
        ξ = chain.hopping
        ξ∞ = asymptotic_hopping(WilsonLog(Λ))               # (1+Λ⁻¹)/2

        # ---- P1 non-vacuity: a genuine logarithmic grid, not a constant stub ----
        @test ξ[1] < 0.95 * ξ∞                              # ξ₀ sits well below the asymptote
        @test ξ[8] > ξ[1]                                   # the grid actually ramps up
        @test ξ∞ ≈ (1 + 1 / Λ) / 2

        # ---- structure of the closed form [KWW1980 ∧ BullaCostiPruschke2008] ----
        @test all(>(0), ξ)                                  # positive hoppings
        @test all(<(ξ∞ + 1e-12), ξ)                         # bounded above by the asymptote
        @test all(≥(0), diff(ξ))                            # non-decreasing, saturating at ξ∞
        @test ξ[end] ≈ ξ∞ rtol = 1e-9                       # converged to the analytic limit

        # ---- εₙ = 0 on the symmetric flat band ----
        @test all(iszero, chain.onsite)
    end

    # ---- universality: ξₙ is independent of the band width D (only the scale is) ----
    c1 = wilson_chain(WilsonLog(2.0), AndersonModel(; U=0.0, Γ=0.1, D=1.0), 40)
    c2 = wilson_chain(WilsonLog(2.0), AndersonModel(; U=0.0, Γ=9.9, D=7.3), 40)
    @test c1.hopping ≈ c2.hopping
end
