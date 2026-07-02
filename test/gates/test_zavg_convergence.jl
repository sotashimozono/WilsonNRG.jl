# Reference-grade CONVERGENCE gate — a trustworthy spectral reference must SYSTEMATICALLY approach the
# exact answer as the z-averaging is refined (more z twists interleave the log grid → finer effective
# resolution). This encodes that law against the one cheaply-exact anchor available at CI scale: the
# U=0 resonant level, whose A(ω) is the exact Lorentzian. NRG is DETERMINISTIC, so the monotone L1-error
# decrease below is an exact, non-flaky invariant. (The interacting strong-coupling unitary-limit
# convergence — the ComplexTimeSIAM-relevant one — needs a larger chain / the ED cross-check and rides
# on the fuller issue #58; πΓA(0)=1 at resolvable params is already gated in test_kondo_resonance.jl.)

using WilsonNRG, Test

# small, fast configuration (the convergence TREND is system-size-robust)
function _alg()
    return NRGAlgorithm(;
        discretization=ZitkoPruschke(2.0), symmetry=U1U1(), truncation=KeepN(150), nsites=35
    )
end
_∫A(r) = sum(r.A[k] * (r.ω[k + 1] - r.ω[k]) for k in 1:(length(r.ω) - 1))

@testset "reference convergence · z-averaging → exact spectral function (issue #58)" begin

    # ---- (a) U=0: the EXACT answer is the Lorentzian A(ω)=(Γ/π)/(ω²+Γ²); the z-averaged spectrum's
    #          L1 error against it must DECREASE as the number of z twists grows — the convergence law. ----
    @testset "U=0 → exact Lorentzian: L1 error ↓ with nz" begin
        Γ = 0.1
        m0 = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
        ωg = collect(-1.0:0.01:1.0)
        Aex = (Γ / π) ./ (ωg .^ 2 .+ Γ^2)
        L1(nz) =
            sum(abs.(zavg_spectral(CFS(), m0, _alg(); nz, b=0.3, ω=ωg).A .- Aex)) /
            length(ωg)
        e1, e4, e8 = L1(1), L1(4), L1(8)
        @test e4 < e1               # z-averaging beats single-z
        @test e8 < e4               # more z-twists ⇒ strictly closer to exact (the convergence law)
        @test e8 < 0.9 * e1         # a real, cumulative gain toward the exact Lorentzian
    end

    # ---- (b) the reference conserves spectral weight: ∫A dω ≈ 1 (interacting, symmetric) ----
    @testset "∫A ≈ 1: weight conservation (U=1)" begin
        m = AndersonModel(; U=1.0, εd=-0.5, Γ=0.1, D=1.0)
        s8 = _∫A(zavg_spectral(CFS(), m, _alg(); nz=8, b=0.3))
        @test 0.9 < s8 < 1.15
    end

    # ---- (c) unit: spectral_at_zero is the window MEAN, robust to a single grid point landing in a
    #          z-interleaved VALLEY (a bare A[argmin|ω|] would read the ~0 there and mislead). ----
    @testset "spectral_at_zero is spike-robust (window mean, not one grid point)" begin
        ω = collect(-0.30:0.02:0.30)
        A = fill(2.0, length(ω))
        A[argmin(abs.(ω))] = 0.0                 # a valley EXACTLY at ω=0 (the misleading single point)
        a0 = spectral_at_zero(ω, A; window=0.1)  # window mean over |ω|<0.1 ≈ 2·(8/9) ≈ 1.78
        @test a0 > 1.5                           # recovers the O(2) window weight, NOT the 0 at ω=0
    end
end
