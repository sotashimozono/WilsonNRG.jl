# ===========================================================================
#  NRG *is* a matrix product state (Saberi, Weichselbaum & von Delft, PRB 78, 035124 (2008)).
#  The kept-eigenvector isometries of the iterative diagonalization,
#      A^{[n]} : (kept_{n-1} ⊗ site_n) → kept_n     (a rectangular, column-orthonormal map),
#  ARE the tensors of a left-canonical MPS: stacking them reconstructs each NRG eigenstate as a
#  vector in the full Fock space. `nrg_mps` returns these isometries (the compact MPS — always
#  available); `reconstruct_mps` contracts them into full-Fock-space vectors for SMALL chains only
#  (contracting is exponential — the whole point of NRG/MPS is to avoid it). At keep-all the
#  reconstruction is exact; the left-canonical property `Φ'Φ = I` and the physical ground state
#  (correct charge/spin sector, ⟨n_d⟩ at the symmetric point) are the faithfulness gate
#  (test/gates/test_nrg_mps.jl). This is the Stage-1 structural half of Saberi's comparison; the
#  variational (vDMRG-vs-NRG) energy comparison is the dynamical half.
# ===========================================================================

# impurity local-basis index for the Anderson impurity: |0⟩=1,|↑⟩=2,|↓⟩=3,|↑↓⟩=4 (engine `_LOC`).
const _IMP_LOCAL = Dict((0, 0) => 1, (1, 1) => 2, (1, -1) => 3, (2, 0) => 4)
const _LOCAL_Q = (0, 1, 1, 2)                                    # charge of |0⟩,|↑⟩,|↓⟩,|↑↓⟩

"per-shell MPS tensor: the kept isometry + its product-basis segmentation and block labels."
struct _ShellIso
    vecs::Dict{NTuple{2,Int},Matrix{Float64}}   # kept eigenvectors per (Q,2Sz) block = the isometry
    seg::Dict{NTuple{2,Int},Vector{_Seg}}       # enlarged-basis segmentation (parent, site, range)
    plan::Dict{NTuple{2,Int},Vector{Int}}       # kept indices per block
end

"""
    nrg_mps(model::AndersonModel, alg) -> (; shells, E)

The NRG flow expressed as a left-canonical MPS (Saberi 2008): `shells[n]` holds the kept-eigenvector
isometry `A^{[n]}` (per `(Q,2Sz)` block), and `E` is the final kept spectrum. `U1U1` only. Contract
with [`reconstruct_mps`](@ref) to obtain the eigenstates as full-Fock vectors (small chains only).
"""
function nrg_mps(model::AndersonModel, alg::NRGAlgorithm)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("nrg_mps needs U1U1 (got $(typeof(alg.symmetry)))"))
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sqrtΛ = sqrt(alg.discretization.Λ)
    st = impurity_init(model, U1U1(), chain)
    shells = _ShellIso[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        plan = truncation_plan(diag.vals, alg.truncation, U1U1())
        kept = Dict(qn => diag.vecs[qn][:, idx] for (qn, idx) in plan)
        push!(shells, _ShellIso(kept, deepcopy(diag.seg), deepcopy(plan)))
        st = update_operators(diag, plan, U1U1())
    end
    return (; shells=shells, E=st.E)
end

"""
    reconstruct_mps(mps; nsites) -> (; states, tags, gnd)

Contract the NRG-MPS isometries into full-Fock-space vectors — a `4^{nsites+1} × n_kept` matrix
`states` whose columns are the reconstructed kept eigenstates (`tags[j] = (block, index)`), plus the
impurity occupation `gnd = ⟨n_d⟩` of the reconstructed ground state. EXPONENTIAL in `nsites` — for
small chains / gates only. At keep-all the reconstruction is exact and `states` is orthonormal.
"""
function reconstruct_mps(mps; nsites::Integer)
    # seed: the impurity is the leftmost (most-significant) 4-dim site, one unit vector per block
    Ψ = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (qn, i) in _IMP_LOCAL
        Ψ[qn] = reshape(Float64[k == i ? 1.0 : 0.0 for k in 1:4], 4, 1)
    end
    fulldim = 4
    for sh in mps.shells
        newfull = 4 * fulldim
        Ψnew = Dict{NTuple{2,Int},Matrix{Float64}}()
        for (nqn, V) in sh.vecs
            M = zeros(newfull, size(V, 2))
            for (pqn, s, r) in sh.seg[nqn]
                haskey(Ψ, pqn) || continue
                Ψp = Ψ[pqn]
                for (a, row) in enumerate(r)
                    @views M[s:4:(newfull - 4 + s), :] .+= Ψp[:, a] * transpose(V[row, :])
                end
            end
            Ψnew[nqn] = M
        end
        Ψ = Ψnew
        fulldim = newfull
    end
    cols = Vector{Float64}[]
    tags = Tuple{NTuple{2,Int},Int}[]
    for (qn, M) in Ψ, j in 1:size(M, 2)
        push!(cols, M[:, j])
        push!(tags, (qn, j))
    end
    states = reduce(hcat, cols)
    # ground state = lowest final energy; its impurity occupation ⟨n_d⟩
    gE = minimum(minimum(v) for v in values(mps.E))
    gqn = first(qn for (qn, v) in mps.E if minimum(v) ≈ gE)
    gvec = Ψ[gqn][:, argmin(mps.E[gqn])]
    gnd = 0.0
    for idx in eachindex(gvec)
        gnd += abs2(gvec[idx]) * _LOCAL_Q[div(idx - 1, 4 ^ nsites) + 1]
    end
    return (; states=states, tags=tags, gnd=gnd, gqn=gqn)
end

# ===========================================================================
#  Stage 2 — the variational half of Saberi 2008: NRG's single forward sweep is a NON-optimal
#  bond-dimension-χ MPS; the variationally-optimal MPS (what vDMRG converges to) has strictly lower
#  energy at the same χ, and both converge to the exact ground energy as χ grows. Verified against
#  the FULL Wilson-chain Hamiltonian built INDEPENDENTLY of the engine (Jordan–Wigner fermions).
# ===========================================================================

using LinearAlgebra: kron, svd, Diagonal, norm, dot, I

# single-orbital ops (basis |0⟩,|↑⟩,|↓⟩,|↑↓⟩; ↑-before-↓ Jordan–Wigner sign on the ↓ operator)
const _CDU = [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]          # c†_↑
const _CDD = [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]         # c†_↓
const _PAR = [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]        # (−1)^n̂ JW string

# single-orbital operator `o` on orbital `i` (0-based) with a JW string on orbitals 0..i-1
function _jw(o, i, L)
    op = fill(1.0, 1, 1)
    for j in 0:(L - 1)
        op = kron(op, j < i ? _PAR : (j == i ? o : Matrix{Float64}(I, 4, 4)))
    end
    return op
end

"""
    wilson_chain_hamiltonian(model::AndersonModel, alg) -> Matrix

The FULL many-body Wilson-chain Hamiltonian `H = Σ_{ij,σ} m_ij c†_{iσ}c_{jσ} + U_eff·n_{d↑}n_{d↓}`,
in the SAME Fock basis as [`reconstruct_mps`](@ref) (impurity = orbital 0, leftmost) but built
INDEPENDENTLY of the engine via explicit Jordan–Wigner fermions. `m` is the rescaled single-particle
chain matrix and `U_eff = U·Λ^{(nsites−1)/2}` matches the engine's per-shell √Λ rescaling, so the NRG
keep-all ground state (`reconstruct_mps`) is `H`'s EXACT ground eigenvector — a fully independent
check of the whole engine, and the reference for the vDMRG-vs-NRG comparison. EXPONENTIAL — small
chains only. `U1U1`.
"""
function wilson_chain_hamiltonian(model::AndersonModel, alg::NRGAlgorithm)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("wilson_chain_hamiltonian needs U1U1"))
    Λ = alg.discretization.Λ
    nsites = alg.nsites
    L = nsites + 1
    chain = wilson_chain(alg.discretization, model, nsites)
    m = reshape([model.εd], 1, 1)                            # rescaled single-particle chain matrix
    for n in 0:(nsites - 1)
        c = n == 0 ? bath_coupling(model) : chain.hopping[n]
        r = n == 0 ? 1.0 : sqrt(Λ)
        k = size(m, 1)
        mn = zeros(k + 1, k + 1)
        mn[1:k, 1:k] = r .* m
        mn[k + 1, k + 1] = chain.onsite[n + 1]
        mn[k, k + 1] = c
        mn[k + 1, k] = c
        m = mn
    end
    cdu = [_jw(_CDU, i, L) for i in 0:(L - 1)]
    cdd = [_jw(_CDD, i, L) for i in 0:(L - 1)]
    H = zeros(4^L, 4^L)
    for i in 1:L, j in 1:L
        m[i, j] == 0.0 && continue
        H .+= m[i, j] .* (cdu[i] * cdu[j]' .+ cdd[i] * cdd[j]')
    end
    H .+=
        (model.U * Λ^((nsites - 1) / 2)) .*
        (_jw(_CDU * _CDU', 0, L) * _jw(_CDD * _CDD', 0, L))
    return H
end

"""
    best_mps_energy(ψ, H, L; D) -> Float64

`⟨ψ_D|H|ψ_D⟩`, where `ψ_D` is the best bond-dimension-`D` MPS approximation of the full state `ψ`
(sequential SVD compression). This is an UPPER bound on the variational (vDMRG) optimum at bond `D`
— so `best_mps_energy < E_NRG` already proves the variationally-optimal MPS (vDMRG) beats NRG's
forward sweep at the same bond dimension (Saberi 2008). Small chains only.
"""
function best_mps_energy(ψ, H, L::Integer; D::Integer)
    tens = Array{Float64,3}[]
    M = reshape(copy(ψ), 1, length(ψ))
    lb = 1
    for i in 1:(L - 1)
        F = svd(reshape(M, lb * 4, size(M, 2) ÷ 4))
        χ = min(D, length(F.S))
        push!(tens, reshape(F.U[:, 1:χ], lb, 4, χ))
        M = Diagonal(F.S[1:χ]) * F.V[:, 1:χ]'
        lb = χ
    end
    push!(tens, reshape(M, lb, 4, 1))
    ψD = reshape(tens[1], 4, size(tens[1], 3))
    for i in 2:L
        χl = size(tens[i], 1)
        χr = size(tens[i], 3)
        ψD = reshape(reshape(ψD, :, χl) * reshape(tens[i], χl, 4 * χr), :, χr)
    end
    v = vec(ψD)
    v ./= norm(v)
    return dot(v, H * v)
end
