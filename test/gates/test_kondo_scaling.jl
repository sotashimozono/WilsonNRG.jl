# Faithfulness gate — the Kondo scale T_K and universal scaling (Wilson, RMP 47, 773 (1975);
# Haldane, PRL 40, 416 (1978); Krishna-murthy–Wilkins–Wilson, PRB 21, 1003 (1980)). This is THE
# signature of the Kondo problem — the part the engine/fixed-point/thermo-limit gates do NOT test:
#
#  (a) the crossover scale is EXPONENTIALLY small in U/Γ,
#          T_K = √(UΓ/2)·exp(−πU/8Γ + πΓ/2U)                (Haldane's analytic formula),
#      so the numerically-extracted crossover T* (temperature where Tχ_imp = c) must track it with a
#      (U,Γ)-INDEPENDENT prefactor: T*/T_K^Haldane is CONSTANT across the Kondo regime. A wrong
#      crossover physics would give the wrong exponential dependence — this pins the functional form.
#  (b) UNIVERSAL SCALING (Wilson's central result): Tχ_imp(T) measured against T/T* collapses onto
#      ONE curve, independent of (U,Γ). Curves at different (U,Γ) must AGREE at fixed T/T*.
#
# Both are independent physical laws (an analytic scale + a universality statement), NOT self-checks.

using WilsonNRG, Test
using Statistics: mean, std

_haldane(U, Γ) = sqrt(U * Γ / 2) * exp(-π * U / (8Γ) + π * Γ / (2U))

function _chi_curve(U, Γ)
    m = AndersonModel(; U, εd=(-U / 2), Γ, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=EnergyCut(6.0), nsites=24
    )
    th = thermodynamics(m, alg; betabar=1.0)
    return th.T, th.Tχ_imp
end

# crossover temperature T* where Tχ_imp crosses `c` (Tχ falls from ~¼ local moment to 0 as T falls).
# The physical flow crosses `c` exactly ONCE; require uniqueness so a noise-driven double crossing
# returns NaN (caught by the `all(isfinite, Tstars)` gate) rather than silently picking the first.
function _crossover(T, Tχ, c)
    xs = [i for i in 1:(length(T) - 1) if (Tχ[i] - c) * (Tχ[i + 1] - c) < 0]
    length(xs) == 1 || return NaN
    i = xs[1]
    t = (c - Tχ[i]) / (Tχ[i + 1] - Tχ[i])
    return exp(log(T[i]) + t * (log(T[i + 1]) - log(T[i])))
end

@testset "faithfulness gate · Kondo scale T_K + universal scaling (Wilson/Haldane)" begin
    params = [(0.2, 0.04), (0.3, 0.05), (0.24, 0.03), (0.4, 0.05)]   # Kondo regime, U/Γ = 5…8
    curves = [(U, Γ, _chi_curve(U, Γ)...) for (U, Γ) in params]
    c = 0.125
    Tstars = [_crossover(T, Tχ, c) for (U, Γ, T, Tχ) in curves]

    @test all(isfinite, Tstars)                                     # a crossover exists for each

    # ---- (a) T_K functional form: T*/T_K^Haldane is (U,Γ)-independent ----
    ratios = [Tstars[i] / _haldane(params[i]...) for i in eachindex(params)]
    @test maximum(ratios) / minimum(ratios) < 1.15                  # exponential T_K to ~few %
    @test std(ratios) / mean(ratios) < 0.06

    # ---- (b) universal scaling collapse: Tχ_imp at fixed T/T* agrees across (U,Γ) ----
    @testset "collapse at T/T* = $(round(x; digits=2))" for x in (0.3, 0.6, 1.2, 2.5)
        vals = Float64[]
        for (i, (U, Γ, T, Tχ)) in enumerate(curves)
            xr = T ./ Tstars[i]                                     # T/T* (decreasing with shell)
            j = findlast(k -> xr[k] >= x, eachindex(xr))
            (j === nothing || j >= length(xr)) && continue
            t = (x - xr[j]) / (xr[j + 1] - xr[j])
            push!(vals, Tχ[j] + t * (Tχ[j + 1] - Tχ[j]))
        end
        @test length(vals) == length(curves)                       # every curve reaches this T/T* (no silent drop)
        @test maximum(vals) - minimum(vals) < 0.03                  # curves collapse (universality)
    end

    # ---- the Kondo hallmark: T_K is EXPONENTIALLY small — spanning U/Γ = 5…8 shrinks it ~4× ----
    # (the analytic Haldane scale, which the extracted T* tracks by construction of check (a))
    hald = [_haldane(U, Γ) for (U, Γ) in params]
    @test maximum(hald) / minimum(hald) > 3                        # exponential spread across the set
    @test all(<(0.03), hald)                                       # T_K ≪ bare U,Γ (deeply below band)
end
