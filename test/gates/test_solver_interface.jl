# Contract gate — the reusable impurity-solver seam (src/solver.jl). `solve(problem, solver)` must be
# a FAITHFUL, DRY wrapper over the existing dispatch: its (G, Σ) equal the direct
# `self_energy` → `_green_from_self_energy` composition byte-for-byte (the independent check, not a
# tautology), `spectral_function` is DERIVED from G, and `n` matches `occupation`. Also gates the
# output contract (Σ/n optional; NO solver type-parameter so heterogeneous collections stay
# concrete), the CommonSolve `init`/`solve!` shape, ω-grid pass-through, and the open-contract
# refusals (clean EngineUnimplemented, never a MethodError) — including SU2SU2+trick, the gap the
# seam's docstring promises won't leak.

using WilsonNRG, Test

@testset "solver interface · solve / impurity_solve contract" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(400), nsites=45
    )
    solver = NRGSolver(;
        algorithm=alg, spectral_method=BHP(), self_energy_method=SelfEnergyTrick()
    )
    sol = solve(ImpurityProblem(m), solver)

    # ---- output contract ----
    @test sol isa ImpuritySolution
    @test length(sol.ω) == length(sol.G) == length(sol.Σ) > 0
    @test sol.Σ isa Vector{ComplexF64}                        # NRG produces Σ
    @test sol.n isa Float64                                   # occupation loaded (U1U1)
    @test spectral_function(sol) ≈ (-1 / π) .* imag.(sol.G)   # A is DERIVED, never stored

    # ---- DRY faithfulness: the seam EQUALS the direct calls, byte-for-byte ----
    se = self_energy(BHP(), m, alg; via=SelfEnergyTrick())
    g = improved_green_function(BHP(), m, alg; via=SelfEnergyTrick())
    @test sol.ω == se.ω
    @test sol.Σ == se.Σ
    @test sol.G == g.G                                        # shares _green_from_self_energy
    @test sol.n ≈ occupation(m, alg).total

    # ---- Fermi-liquid pin survives the wrapper (the physics the seam must transport) ----
    i0 = argmin(abs.(sol.ω))
    @test isapprox(real(sol.Σ[i0]), m.U / 2; atol=0.05)       # ReΣ(0)=U/2 at the symmetric point

    # ---- ω-grid pass-through (the contract's grid obligation) ----
    ωc = collect(range(-0.3, 0.3; length=51))
    solc = solve(ImpurityProblem(m), solver; ω=ωc, with_occupation=false)
    @test solc.ω == ωc
    @test solc.n === nothing                                  # occupation skipped ⇒ nothing (optional field)

    # ---- output type is NOT parametrized on the solver: heterogeneous collections stay concrete ----
    @test eltype([sol, solc]) === ImpuritySolution
    @test isconcretetype(eltype([sol, solc]))

    # ---- CommonSolve init/solve! shape (the DMFT warm-restart entry) ----
    cache = init(ImpurityProblem(m), solver; with_occupation=false)
    @test solve!(cache).G == sol.G

    # ---- convenience aliases route to the same seam ----
    @test impurity_solve(solver, m).G == sol.G
    @test hasmethod(impurity_solve, Tuple{AndersonModel})     # `impurity_solve(model)` default-solver shortcut

    # ---- defaults + the placeholder eigensolver axis ----
    d = NRGSolver()
    @test d isa AbstractImpuritySolver
    @test d.eigensolver == DenseEigen()
    @test d.spectral_method == default_spectral_method()
    @test ImpurityProblem(m).bath == FlatBand()               # default bath

    # ---- open-contract refusals: clean EngineUnimplemented, never a MethodError ----
    @test_throws EngineUnimplemented solve(
        ImpurityProblem(KondoModel(; J=0.1)), NRGSolver()
    )
    @test_throws EngineUnimplemented solve(                   # NumericalBath ⇒ needs the general-DOS chain
        ImpurityProblem(m, NumericalBath([0.0], [0.0 + 0.0im])),
        NRGSolver(),
    )
    @test_throws EngineUnimplemented solve(                   # SU2SU2 + trick: refuses, does NOT MethodError
        ImpurityProblem(m),
        NRGSolver(;
            algorithm=NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=SU2SU2(),
                truncation=KeepN(50),
                nsites=4,
            ),
            self_energy_method=SelfEnergyTrick(),
        ),
    )
end
