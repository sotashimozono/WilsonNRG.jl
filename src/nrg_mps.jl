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
