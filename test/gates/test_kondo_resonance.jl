# Faithfulness gate — the Kondo resonance in A(ω) at U>0. THE spectroscopic signature of the Kondo
# effect: the symmetric Anderson model has a sharp Kondo (Abrikosov–Suhl) peak at ω=0 pinned to the
# UNITARY LIMIT
#       πΓ A(0) = sin²(π n_d / 2) = 1                    (half-filling; Friedel/Langreth sum rule),
# dominating the Hubbard side-bands at ω ≈ ±U/2, with a width ~T_K ≪ U. The self-energy-improved
# spectral function `A(ω) = -Im[1/(ω-εd-Δ-Σ)]/π` resolves it — the standard accurate NRG route
# (Bulla, RMP 80, 395 (2008)) — because the Fermi-liquid pins ReΣ(0)=U/2, ImΣ(0)=0 tie A(0) to the
# unitary limit exactly, whereas the DIRECT log-Gaussian spectral function washes the narrow Kondo
# peak out. The target πΓA(0)=1 is an INDEPENDENT physical law (the Friedel sum rule at half-filling),
# not a self-check.

using WilsonNRG, Test
using Statistics: median

# Robust value of A at ω→0: the median of A over the grid points NEAREST ω=0. Those points all lie
# at |ω| ≪ T_K, so A is Fermi-liquid-flat there (A ≈ A(0)); the median rejects the 1–2-point spike the
# self-energy trick's Σ produces right AT ω=0 (a Re G=0 crossing makes Σ=U·F/G jump for a single grid
# step), which a bare `argmin(|ω|)` could otherwise land on — making the gate depend on a
# floating-point tie-break rather than the physics.
_A0(A, ω) = median(A[partialsortperm(abs.(ω), 1:9)])

@testset "faithfulness gate · Kondo resonance πΓA(0)=1 (unitary limit)" begin
    Γ = 0.05

    @testset "unitary-limit peak · U=$U" for U in (0.0, 0.3, 0.5)
        m = AndersonModel(; U, εd=(-U / 2), Γ, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(400), nsites=45
        )
        g = improved_green_function(BHP(), m, alg)
        A = (-1 / π) .* imag.(g.G)
        ω = g.ω
        A0 = _A0(A, ω)                                       # robust unitary-limit height
        @test 0.8 < π * Γ * A0 < 1.15                        # Friedel unitary limit πΓA(0)=1
        @test abs(ω[argmax(A)]) < Γ                          # the peak sits at ω≈0 (Kondo, not a band)
        if U > 0                                             # the Kondo peak dominates the Hubbard bands
            ihub = argmin(abs.(ω .- U / 2))
            @test A0 > 1.5 * A[ihub]
        end
        # (p-h symmetry of the improved A is NOT asserted here: it inherits BHP's windowed p-h break
        #  at U>0 — the documented @test_broken of test_spectral_sumrules.jl / issue #33 — which the
        #  height/structure checks above are independent of.)
    end

    # U=0 is the exact resonant level (Σ≡0 ⇒ improved G = G₀): πΓA(0)=1 to high accuracy
    @testset "U=0 resonant level exact" begin
        m0 = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(400), nsites=45
        )
        g = improved_green_function(BHP(), m0, alg)
        A = (-1 / π) .* imag.(g.G)
        @test isapprox(π * Γ * _A0(A, g.ω), 1.0; atol=0.05)
    end
end
