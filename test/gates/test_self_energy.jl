# Faithfulness gate — impurity self-energy Σ(ω), method-comparison platform (Axis 4b).
# Diverse methods exist because Σ is accuracy-sensitive; the gate pins the ROBUST default
# (the self-energy trick Σ=U·F/G) against exact relations, and demonstrates that the methods
# differ in robustness (the reason for the comparison platform):
#  (1) Fermi liquid at the symmetric point: ReΣ(0)=U/2, ImΣ(0)=0, and ReΣ(ω)+ReΣ(−ω)=U;
#  (2) U=0 ⇒ Σ=0 EXACTLY for the trick (Σ∝U), while Dyson carries the broadening error;
#  (3) compare_self_energy quantifies the cross-method disagreement.
# Refs: Bulla–Hewson–Pruschke, PRB 57, 10287 (1998); Bulla–Costi–Pruschke, RMP 80, 395 (2008).

using WilsonNRG, Test

@testset "method-recovery gate · self-energy Σ(ω) [trick vs Dyson]" begin
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(300), nsites=26
    )
    near(ωs, x) = argmin(abs.(ωs .- x))

    # ---- (1) trick at U>0 symmetric point: Fermi-liquid / Friedel ----
    @testset "trick: ReΣ(0)=U/2, ImΣ(0)=0 (Fermi liquid)" begin
        U = 0.5
        m = AndersonModel(; U, εd=-U / 2, Γ=0.1, D=1.0)
        se = self_energy(m, alg)                       # default = BHP green + SelfEnergyTrick
        ip = near(se.ω, 0.02)
        im_ = near(se.ω, -0.02)
        @test isapprox(real(se.Σ[ip]), U / 2; atol=0.06)            # ReΣ(0) = U/2 (Friedel)
        @test abs(imag(se.Σ[ip])) < 0.06                              # ImΣ(0) = 0 (Fermi liquid)
        @test isapprox(real(se.Σ[ip]) + real(se.Σ[im_]), U; atol=0.08)  # p-h: ReΣ(ω)+ReΣ(−ω)=U
    end

    # ---- (2) U=0 ⇒ trick is EXACTLY 0 (Σ∝U); Dyson is broadening-noisy ----
    @testset "U=0: trick exact 0, Dyson noisy (robustness gap)" begin
        m0 = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D=1.0)
        cmp = compare_self_energy(m0, alg)                            # runs trick + Dyson
        @test maximum(abs, cmp.Σ[:SelfEnergyTrick]) < 1.0e-9          # exact 0 (no broadening error)
        reg = findall(x -> 0.03 < abs(x) < 0.5, cmp.ω)
        @test maximum(abs, cmp.Σ[:Dyson][reg]) > 0.01                 # Dyson visibly nonzero
        @test cmp.disagreement > 0.05                                # methods disagree ⇒ comparison is meaningful
    end

    # ---- green_function default form returns a complex retarded G ----
    @testset "green_function default form" begin
        m = AndersonModel(; U=0.3, εd=-0.15, Γ=0.1, D=1.0)
        gf = green_function(m, alg)
        @test eltype(gf.G) <: Complex
        @test all(≤(0), imag.(gf.G))                                 # Im G ≤ 0 (retarded; A=-ImG/π ≥ 0)
        @test length(gf.G) == length(gf.ω)
    end
end
