# Faithfulness gate — z-averaging (Oliveira–Oliveira 1994; Žitko–Pruschke, PRB 79, 085106 (2009))
# improves the spectral resolution. Averaging A(ω) over discretization twists z ∈ (0,1] interleaves
# the log-grid poles, so the artifacts of any single z cancel and A(ω) converges toward the exact
# continuum spectrum. The INDEPENDENT target is the U=0 resonant level A(ω) = (Γ/π)/(ω²+Γ²) (exact,
# closed form): the z-averaged A is strictly CLOSER to it — in integrated L1 error AND in the peak
# height A(0)=1/(πΓ) — than the single-z result, while the sum rule ∫A=1 is preserved. This is not
# a self-check: it is measured against the analytic Lorentzian.

using WilsonNRG, Test

@testset "faithfulness gate · z-averaging improves spectral resolution (vs U=0 Lorentzian)" begin
    Γ = 0.05
    Λ = 2.5
    m = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
    lor(w) = (Γ / π) / (w^2 + Γ^2)                         # exact U=0 resonant level

    alg = NRGAlgorithm(;
        discretization=ZitkoPruschke(Λ; z=1.0),
        symmetry=U1U1(),
        truncation=KeepN(300),
        nsites=30,
    )
    g1 = green_function(CFS(), m, alg)
    ω = g1.ω
    A1 = (-1 / π) .* imag.(g1.G)                            # single z=1
    gz = zavg_green_function(CFS(), m, alg; nz=16)
    Az = (-1 / π) .* imag.(gz.G)                            # z-averaged

    @test gz.ω == g1.ω
    Alor = [lor(w) for w in ω]
    dω = [i == 1 ? ω[2] - ω[1] : ω[i] - ω[i - 1] for i in eachindex(ω)]
    l1(A) = sum(abs.(A .- Alor) .* abs.(dω))
    trapz(A) = sum((A[i] + A[i + 1]) / 2 * (ω[i + 1] - ω[i]) for i in 1:(length(ω) - 1))
    k0 = argmin(abs.(ω))

    @test l1(Az) < l1(A1)                                  # integrated shape strictly closer to exact
    @test abs(Az[k0] - lor(0.0)) < abs(A1[k0] - lor(0.0))  # peak height closer to 1/(πΓ)
    @test 0.9 < trapz(Az) < 1.1                            # sum rule preserved by the average

    # z-averaging is a linear average ⇒ nz=1 reproduces the single-z result exactly
    @test maximum(
        abs, (-1 / π) .* imag.(zavg_green_function(CFS(), m, alg; nz=1).G) .- A1
    ) < 1e-12

    # honest refusal: the conventional (z-less) WilsonLog cannot be z-averaged
    algw = NRGAlgorithm(;
        discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(64), nsites=6
    )
    @test_throws EngineUnimplemented zavg_green_function(CFS(), m, algw)
end
