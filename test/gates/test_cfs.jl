# Faithfulness gate — complete-Fock-space (CFS) spectral function A(ω) at T=0.
# CFS sums the Lehmann representation over the complete basis of discarded states, so the
# spectral sum rule ∫A dω = 1 holds EXACTLY by completeness — unlike the BHP windowed
# patching, which only approximates it. The gate checks: (1) the sum rule is tight AND
# strictly tighter than BHP on the *same* flow (the comparison-platform payoff);
# (2) particle–hole symmetry A(ω)=A(−ω); (3) the resonance at the hybridization scale;
# (4) non-negativity. At U=0 the impurity is the resonant level A(ω)=(Γ/π)/(ω²+Γ²).
# Refs: Peters–Pruschke–Anders, PRB 74, 245114 (2006); Anders–Schiller, PRL 95, 196801 (2005).

using WilsonNRG, Test

@testset "method-recovery gate · CFS spectral function A(ω) [T=0, U=0]" begin
    Γ = 0.1
    m = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(300), nsites=26
    )
    res = spectral(CFS(), m, alg)
    ω, A = res.ω, res.A
    @test all(≥(0), A)                                          # spectral function is non-negative

    trapz(x, y) = sum((y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i]) for i in 1:(length(x) - 1))

    # ---- (1) sum rule ∫A dω = 1 — EXACT by completeness, and tighter than BHP ----
    ∫A_cfs = trapz(ω, A)
    @test isapprox(∫A_cfs, 1.0; atol=0.03)                      # ≈1 (pole weights sum to 1.00006)
    A_bhp = spectral(BHP(), m, alg; ω=ω).A                      # same flow + grid, for comparison
    ∫A_bhp = trapz(ω, A_bhp)
    @test abs(∫A_cfs - 1) < abs(∫A_bhp - 1)                     # CFS strictly closer to the exact 1

    # ---- (2) particle–hole symmetry A(ω) = A(−ω) at the symmetric point ----
    npos = length(ω) ÷ 2
    @test maximum(abs, A[(npos + 1):end] .- reverse(A[1:npos])) < 1.0e-3 * maximum(A)  # ~1e-5

    # ---- (3) resonance at the hybridization scale ~Γ, decaying into the band wings ----
    A_at(x) = A[argmin(abs.(ω .- x))]
    @test A_at(Γ) > 1.0                                         # substantial weight at ω~Γ
    @test A_at(Γ) > 5 * A_at(1.0)                               # ≫ band-edge weight (resonance)

    # ---- (4) U>0 interacting regime — the cited methods' actual target (U=0 is free).
    # Sum rule (completeness, broadening-independent) + p-h symmetry must survive interactions;
    # p-h also guards the degenerate-GS seed (the odd-parity Kondo doublet must be split, not picked).
    @testset "CFS at U>0 (symmetric point)" begin
        U = 0.5
        mU = AndersonModel(; U, εd=-U / 2, Γ, D=1.0)
        rU = spectral(CFS(), mU, alg)
        ωU, AU = rU.ω, rU.A
        @test isapprox(trapz(ωU, AU), 1.0; atol=0.05)           # sum rule survives interactions (≈1.003)
        npU = length(ωU) ÷ 2
        @test maximum(abs, AU[(npU + 1):end] .- reverse(AU[1:npU])) < 1.0e-3 * maximum(AU)  # p-h (≈1e-5)
    end
end
