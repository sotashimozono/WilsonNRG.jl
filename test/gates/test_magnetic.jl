# Faithfulness gate — impurity magnetic response (finite field).
# The field −h·Sz commutes with H (Sz conserved), so it is applied exactly in the
# Boltzmann weights of the zero-field spectrum. Checks:
#  (1) a decoupled impurity is a free spin-½: M_imp(T,h) = ½ tanh(h/2T), saturating to ½.
#  (2) fluctuation–dissipation: the linear response ∂M_imp/∂h = χ_imp reproduces the
#      fluctuation susceptibility T·χ_imp = 1/4 of the thermodynamics gate.
#  (3) spin symmetry: no spontaneous moment, M_imp(h=0) = 0.
# Uses EnergyCut (clean two-run subtraction; see src/thermodynamics.jl).

using WilsonNRG, Test

@testset "method-recovery gate · magnetic response" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0)   # decoupled ⇒ free spin
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=EnergyCut(6.5), nsites=16
    )

    # ---- (1) free-spin Brillouin curve: M_imp = ½ tanh(h/2T), → ½ at T→0 ----
    @testset "free spin: M_imp = ½tanh(h/2T)" begin
        h = 0.05
        mag = magnetization(m, alg; h)
        lowT = findall(<(0.035), mag.T)                  # well into the saturated regime
        @test !isempty(lowT)
        for k in lowT
            @test isapprox(mag.M_imp[k], 0.5 * tanh(h / (2 * mag.T[k])); atol=0.01)
        end
        @test mag.M_imp[end] > 0.49                      # saturates to ½ at the lowest T
        @test all(≤(0.5 + 0.01), mag.M_imp)              # never exceeds the spin-½ maximum
    end

    # ---- (2) fluctuation–dissipation: dM/dh = χ ⇒ T·χ → 1/4 (= thermodynamics value) ----
    @testset "linear response χ matches T·χ = 1/4" begin
        h = 1.0e-4
        mag = magnetization(m, alg; h)
        for k in findall(<(0.035), mag.T)
            @test isapprox(mag.T[k] * mag.M_imp[k] / h, 0.25; atol=0.01)
        end
    end

    # ---- (3) spin symmetry: M_imp(h=0) = 0 ----
    @testset "no spontaneous moment" begin
        small = NRGAlgorithm(;
            discretization=WilsonLog(2.0),
            symmetry=U1U1(),
            truncation=EnergyCut(6.5),
            nsites=8,
        )
        mag = magnetization(m, small; h=0.0)
        @test all(x -> abs(x) < 1.0e-9, mag.M_imp)
    end
end
