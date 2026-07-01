# Genealogy-coverage gate — the research genealogy must BRANCH through the dispatch axes so that
# every implemented method is actually usable, and every unimplemented combination fails HONESTLY
# with EngineUnimplemented, never a raw MethodError/FieldError (the class of hidden "not actually
# usable" gap: z-averaging that threw in nrg_solve; the Kondo × z-averaging FieldError on model.Γ).
# This locks the invariant across  model × discretization × symmetry × {nrg_solve, thermodynamics,
# magnetization, occupation, green_function, self_energy}:  every branch is OK or honestly-unimpl.

using WilsonNRG, Test

# classify one genealogy branch: :ok (runs) | :unimpl (honest) | :gap (raw MethodError/other = a bug)
function _branch(thunk)
    try
        thunk()
        return :ok
    catch e
        e isa WilsonNRG.EngineUnimplemented && return :unimpl
        return :gap                       # MethodError / FieldError / anything else = hidden gap
    end
end

@testset "genealogy-coverage gate · every branch usable or honestly unimplemented" begin
    Γ = 0.05
    D = 1.0
    discs = (WilsonLog(2.5), CampoOliveira(2.5), ZitkoPruschke(2.5))
    And = AndersonModel(; U=0.3, εd=-0.15, Γ, D)
    Kon = KondoModel(; J=0.2, D)
    alg(d, s; tr=KeepN(150), n=6) =
        NRGAlgorithm(; discretization=d, symmetry=s, truncation=tr, nsites=n)

    # ---- (a) INVARIANT: NO hidden gaps anywhere in the cross-product (the load-bearing check) ----
    @testset "no hidden dispatch gaps across the axes" begin
        for m in (And, Kon), d in discs, s in (U1U1(), U1SU2(), SU2SU2())
            @test _branch(() -> nrg_solve(m, alg(d, s))) != :gap
        end
        for s in (U1U1(), U1SU2(), SU2SU2())
            a = alg(WilsonLog(2.5), s; tr=EnergyCut(7.0), n=10)
            @test _branch(() -> thermodynamics(And, a)) != :gap
            @test _branch(() -> magnetization(And, a; h=0.01)) != :gap
        end
        for d in discs, s in (U1U1(), U1SU2(), SU2SU2())
            @test _branch(() -> occupation(And, alg(d, s))) != :gap
            @test _branch(() -> double_occupancy(And, alg(d, s))) != :gap
        end
        for meth in (BHP(), CFS(), FDM(), DMNRG()), d in discs, s in (U1U1(), U1SU2())
            @test _branch(() -> green_function(meth, And, alg(d, s))) != :gap
        end
        for via in (SelfEnergyTrick(), Dyson()), s in (U1U1(), U1SU2())
            @test _branch(() -> self_energy(BHP(), And, alg(WilsonLog(2.5), s); via)) !=
                :gap
        end
        qfin = AndersonModel(; U=0.3, εd=0.1, Γ, D)         # shares the bath with And
        for s in (U1U1(), U1SU2(), SU2SU2())
            @test _branch(
                () -> quench_dynamics(And, qfin, alg(WilsonLog(2.5), s); times=[0.0, 1.0])
            ) != :gap
        end
    end

    # ---- (b) POSITIVE coverage: the implemented branches actually RUN (not just non-throwing) ----
    @testset "implemented branches run across every discretization" begin
        for d in discs
            @test _branch(() -> nrg_solve(And, alg(d, U1U1()))) == :ok
            @test _branch(() -> nrg_solve(And, alg(d, U1SU2()))) == :ok
            @test _branch(() -> nrg_solve(Kon, alg(d, U1U1()))) == :ok      # ← the fixed Kondo×z-avg
            @test _branch(() -> occupation(And, alg(d, U1U1()))) == :ok
            @test _branch(() -> double_occupancy(And, alg(d, U1U1()))) == :ok
            for meth in (BHP(), CFS(), FDM(), DMNRG())
                @test _branch(() -> green_function(meth, And, alg(d, U1U1()))) == :ok
            end
        end
        @test _branch(
            () -> thermodynamics(And, alg(WilsonLog(2.5), U1SU2(); tr=EnergyCut(7.0), n=10))
        ) == :ok
        @test _branch(() -> self_energy(And, alg(WilsonLog(2.5), U1U1()))) == :ok
        @test _branch(
            () -> quench_dynamics(
                And,
                AndersonModel(; U=0.3, εd=0.1, Γ, D),
                alg(WilsonLog(2.5), U1U1());
                times=[0.0, 1.0],
            ),
        ) == :ok
    end

    # ---- (b') SU2SU2 wired for nrg_solve at the p–h symmetric point; honest EngineUnimplemented
    #          off it and for the not-yet-wired SU2SU2 operations (never a raw gap). ----
    @testset "SU2SU2 wired for nrg_solve (symmetric), honest otherwise" begin
        for d in discs
            @test _branch(() -> nrg_solve(And, alg(d, SU2SU2()))) == :ok     # And is p–h symmetric
        end
        asym = AndersonModel(; U=0.3, εd=-0.05, Γ, D)                        # εd ≠ −U/2 = −0.15
        @test _branch(() -> nrg_solve(asym, alg(WilsonLog(2.5), SU2SU2()))) == :unimpl
        @test _branch(() -> nrg_solve(Kon, alg(WilsonLog(2.5), SU2SU2()))) == :unimpl    # Kondo SU2SU2
        @test _branch(() -> thermodynamics(And, alg(WilsonLog(2.5), SU2SU2()))) == :unimpl
        @test _branch(() -> occupation(And, alg(WilsonLog(2.5), SU2SU2()))) == :unimpl
        @test _branch(() -> green_function(BHP(), And, alg(WilsonLog(2.5), SU2SU2()))) ==
            :unimpl
    end

    # ---- (c) faithfulness of the Kondo fix: the z-averaging chain is MODEL-AGNOSTIC (Γ cancels in
    #          the f₀ normalization), so the Kondo band chain is identical to the Anderson one at
    #          equal D — the fix is correct, not merely non-throwing. ----
    @testset "z-averaging chain is model-agnostic (Kondo fix is faithful)" begin
        for Z in (CampoOliveira(2.5), ZitkoPruschke(2.5))
            ca = wilson_chain(Z, AndersonModel(; U=0.0, εd=0.0, Γ=0.05, D=1.0), 15)
            ck = wilson_chain(Z, KondoModel(; J=0.2, D=1.0), 15)
            @test ca.hopping == ck.hopping        # depends on D only — not Γ, not the model type
            @test ck.onsite == zeros(15)
        end
    end
end
