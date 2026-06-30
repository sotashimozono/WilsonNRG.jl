# Faithfulness gate — SU(2) angular-momentum coefficients (foundation for U1SU2).
# These are independently verifiable against textbook values (Edmonds), so they are checked
# directly here; the non-abelian engine's 6j recoupling will instead be pinned against the
# U1U1 spectrum (cross-method agreement) once it lands.

using WilsonNRG, Test
using WilsonNRG: clebsch_gordan, wigner3j, wigner6j

@testset "method-recovery gate · SU(2) CG/6j coefficients" begin
    # ---- Clebsch–Gordan vs known values ----
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, -1 // 2, 0, 0) ≈ 1 / sqrt(2)   # singlet
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, 1 // 2, 1, 1) ≈ 1.0            # stretched
    @test clebsch_gordan(1, 0, 1 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ -1 / sqrt(3)
    @test clebsch_gordan(1 // 2, 1 // 2, 1, 0, 3 // 2, 1 // 2) ≈ sqrt(2 / 3)

    # orthonormality: Σ_{m1} ⟨j1 m1; j2 M-m1|J M⟩² = 1 for each (J,M) in the decomposition
    s = sum(clebsch_gordan(1, m, 1 // 2, 1 // 2 - m, 3 // 2, 1 // 2)^2 for m in (-1, 0, 1))
    @test s ≈ 1.0

    # ---- 6j vs known values ----
    @test wigner6j(1 // 2, 1 // 2, 1, 1 // 2, 1 // 2, 1) ≈ 1 / 6
    @test wigner6j(1, 1, 1, 1, 1, 1) ≈ 1 / 6
    @test wigner6j(1 // 2, 1 // 2, 0, 1 // 2, 1 // 2, 1) ≈ 1 / 2
    @test wigner6j(1, 1, 0, 1, 1, 1) ≈ -1 / 3                                   # {1 1 0;1 1 1}

    # triangle violations vanish
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, 1 // 2, 0, 0) == 0.0           # |1/2-1/2|..=0 only
    @test wigner6j(1 // 2, 1 // 2, 5, 1 // 2, 1 // 2, 1) == 0.0

    # ---- multiplet weighting: 2S+1 ----
    @test WilsonNRG.multiplicity(U1SU2(), (1, 1 // 2)) == 2
    @test WilsonNRG.multiplicity(U1SU2(), (0, 0 // 1)) == 1
    @test WilsonNRG.multiplicity(U1SU2(), (1, 3 // 2)) == 4
end
