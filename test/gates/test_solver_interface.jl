# Contract gate — the reusable impurity-solver seam (src/solver.jl). `impurity_solve` must be a
# FAITHFUL, DRY wrapper over the existing dispatch, NOT a re-derivation: its (G, Σ, A) have to equal
# the direct composition `self_energy` → G = _green_from_self_energy(Σ) → A = -Im G/π to machine
# precision. That equality is the independent check — it proves there is a single computation path
# behind the seam a DMFT/DMET loop (or a cross-solver benchmark) will call. Also gates the defaults
# and the open-contract refusal.

using WilsonNRG, Test

@testset "solver interface · impurity_solve contract" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(400), nsites=45
    )
    solver = NRGSolver(;
        algorithm=alg, spectral_method=BHP(), self_energy_method=SelfEnergyTrick()
    )
    sol = impurity_solve(solver, m)

    # ---- structural: a stable, parallel-array output with A = -Im G/π ----
    @test sol isa ImpuritySolution
    @test length(sol.ω) == length(sol.G) == length(sol.Σ) == length(sol.A) > 0
    @test eltype(sol.G) == ComplexF64 &&
        eltype(sol.Σ) == ComplexF64 &&
        eltype(sol.A) == Float64
    @test sol.A ≈ (-1 / π) .* imag.(sol.G)

    # ---- DRY faithfulness: the seam EQUALS the direct calls (one computation path, no divergence) ----
    se = self_energy(BHP(), m, alg; via=SelfEnergyTrick())
    g = improved_green_function(BHP(), m, alg; via=SelfEnergyTrick())
    @test sol.ω ≈ se.ω
    @test sol.Σ ≈ se.Σ
    @test sol.G ≈ g.G                       # improved_green_function shares _green_from_self_energy

    # ---- Fermi-liquid pin survives the wrapper (the physics the seam must transport) ----
    i0 = argmin(abs.(sol.ω))
    @test isapprox(real(sol.Σ[i0]), m.U / 2; atol=0.05)   # ReΣ(0)=U/2 at the symmetric point

    # ---- defaults & convenience entry (no heavy run: just the contract surface) ----
    d = NRGSolver()
    @test d isa AbstractImpuritySolver
    @test d.spectral_method == default_spectral_method()          # BHP
    @test d.self_energy_method == default_self_energy_method()     # SelfEnergyTrick
    @test d.algorithm.discretization == WilsonLog(2.0)
    @test hasmethod(impurity_solve, Tuple{AndersonModel})         # `impurity_solve(model)` shortcut exists

    # ---- open contract: a model with no self-energy route refuses cleanly (not a MethodError) ----
    @test_throws EngineUnimplemented impurity_solve(KondoModel(; J=0.1))
end
