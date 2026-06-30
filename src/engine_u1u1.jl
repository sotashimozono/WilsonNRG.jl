# ===========================================================================
#  Iterative-diagonalization engine for U(1)charge × U(1)spin (`U1U1`).
#
#  This is the symmetry layer behind the generic `nrg_solve` driver: it provides
#  the methods of the engine seam (`impurity_init` / `add_site` /
#  `diagonalize_blocks` / `update_operators`) for the abelian `(Q, 2Sₙ)` setting,
#  mirroring the `NonHermitianNRG` reference but Hermitian (real symmetric blocks).
#
#  Conventions
#  -----------
#  * Blocks are labelled by `(Q, D)` with `Q` = electron number, `D = 2·Sₙ`.
#  * One electron orbital per site, basis order 1=|0⟩, 2=|↑⟩, 3=|↓⟩, 4=|↑↓⟩,
#    with `|↑↓⟩ ≡ c†↑ c†↓ |0⟩`.
#  * Recursion `H_{N+1} = √Λ · H_N + ξ_N Σ_σ(f†_{Nσ} f_{N+1σ} + h.c.) + ε_{N+1} n_{N+1}`.
#    The `√Λ` multiplies only the kept diagonal energies; hoppings use the bare
#    O(1) coefficient ξ_N. The impurity ⊗ f₀ step is the same recursion with
#    coupling `V₀` and no rescaling — so init and iteration share one code path.
#  * Fermion sign: f_{N+1σ} (rightmost) anticommutes past the kept fermions of the
#    parent ⇒ a factor (−1)^{Q_parent}.
# ===========================================================================

using LinearAlgebra: Symmetric, Diagonal, eigen

# charge and 2·Sz of each single-site state |0⟩,|↑⟩,|↓⟩,|↑↓⟩
const _LOC_Q = (0, 1, 1, 2)
const _LOC_D = (0, 1, -1, 0)

# Annihilation c_σ on the new site, as (σ_d, moves) with moves = ((s_from, s_to, amp), …):
#   c↑: |↑⟩→|0⟩ (+1), |↑↓⟩→|↓⟩ (+1)        c↓: |↓⟩→|0⟩ (+1), |↑↓⟩→|↑⟩ (−1)
const _ANNIHILATE = ((1, ((2, 1, 1.0), (4, 3, 1.0))), (-1, ((3, 1, 1.0), (4, 2, -1.0))))
# Creation c†_σ on the new site (adjoint of the above):
#   c†↑: |0⟩→|↑⟩ (+1), |↓⟩→|↑↓⟩ (+1)       c†↓: |0⟩→|↓⟩ (+1), |↑⟩→|↑↓⟩ (−1)
const _CREATE = ((1, ((1, 2, 1.0), (3, 4, 1.0))), (-1, ((1, 3, 1.0), (2, 4, -1.0))))

"""
    U1U1State

NRG state in the `(Q, D=2Sₙ)` block basis: kept eigen-energies per block and the
matrix elements of the last-site creation operator `f†_σ` between adjacent
charge blocks (the data propagated across iterations).
"""
struct U1U1State
    E::Dict{NTuple{2,Int},Vector{Float64}}      # (Q,D) → kept eigenenergies of H_N
    F::Dict{NTuple{3,Int},Matrix{Float64}}      # (Q,D,σd) → ⟨Q+1,D+σd| f†_σ |Q,D⟩
end

# segment = (parent block, local state s, row range within the enlarged block)
const _Seg = Tuple{NTuple{2,Int},Int,UnitRange{Int}}

"Enlarged (pre-diagonalization) Hamiltonian blocks + their product-basis segmentation."
struct U1U1Enlarged
    H::Dict{NTuple{2,Int},Matrix{Float64}}
    seg::Dict{NTuple{2,Int},Vector{_Seg}}
end

"Per-block eigendecomposition of an enlarged Hamiltonian (segments carried for the operator rebuild)."
struct U1U1Diag
    vals::Dict{NTuple{2,Int},Vector{Float64}}
    vecs::Dict{NTuple{2,Int},Matrix{Float64}}
    seg::Dict{NTuple{2,Int},Vector{_Seg}}
end

# ---- the impurity (iteration −1): a trivial 4-state NRG state ----
"""
    bath_coupling(model::AndersonModel) -> Float64

Impurity↔f₀ hybridization amplitude `V₀ = √(2 D Γ / π)` for a flat band
(`V₀² = ∫dω Γ/π`). CONVENTION: the `WilsonLog` discretization correction factor
`A_Λ` is *not* folded in here; it matters only for absolute spectral accuracy and
is revisited with the spectral layer (Stage 3). The energy-flow / free-fermion
checks are independent of the precise `V₀`.
"""
bath_coupling(model::AndersonModel) = sqrt(2 * model.D * model.Γ / π)

function impurity_init(m::AndersonModel, ::U1U1, ::WilsonChain)
    E = Dict{NTuple{2,Int},Vector{Float64}}(
        (0, 0) => [0.0], (1, 1) => [m.εd], (1, -1) => [m.εd], (2, 0) => [2 * m.εd + m.U]
    )
    F = Dict{NTuple{3,Int},Matrix{Float64}}(
        (0, 0, 1) => fill(1.0, 1, 1),    # c†↑: |0⟩→|↑⟩
        (1, -1, 1) => fill(1.0, 1, 1),    # c†↑: |↓⟩→|↑↓⟩
        (0, 0, -1) => fill(1.0, 1, 1),    # c†↓: |0⟩→|↓⟩
        (1, 1, -1) => fill(-1.0, 1, 1),    # c†↓: |↑⟩→ −|↑↓⟩
    )
    return U1U1State(E, F)
end

# ---- attach one Wilson site (the generic block recursion) ----
function add_site(st::U1U1State, ::U1U1; coupling::Real, rescale::Real, onsite::Real=0.0)
    # 1. group product states |parent; s⟩ into new blocks and lay out segments
    raw = Dict{NTuple{2,Int},Vector{Tuple{NTuple{2,Int},Int}}}()
    for (pqn, Evec) in st.E
        for s in 1:4
            nqn = (pqn[1] + _LOC_Q[s], pqn[2] + _LOC_D[s])
            push!(get!(raw, nqn, Tuple{NTuple{2,Int},Int}[]), (pqn, s))
        end
    end
    seg = Dict{NTuple{2,Int},Vector{_Seg}}()
    for (nqn, pairs) in raw
        sort!(pairs)                       # stable, reproducible segment order
        segs = _Seg[]
        off = 0
        for (pqn, s) in pairs
            d = length(st.E[pqn])
            push!(segs, (pqn, s, (off + 1):(off + d)))
            off += d
        end
        seg[nqn] = segs
    end

    # 2. assemble each block: diagonal (rescaled energies + on-site) + hopping T + Tᵀ
    H = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (nqn, segs) in seg
        dim = isempty(segs) ? 0 : last(segs[end][3])
        diagv = zeros(Float64, dim)
        T = zeros(Float64, dim, dim)
        rng = Dict((pqn, s) => r for (pqn, s, r) in segs)   # (parent,s) → range
        for (pqn, s, r) in segs
            for (k, i) in enumerate(r)
                diagv[i] = rescale * st.E[pqn][k] + onsite * _LOC_Q[s]
            end
        end
        for (pqn, sB, rB) in segs               # source segment |pqn; sB⟩
            QB, DB = pqn
            for (σd, moves) in _ANNIHILATE
                Fkey = (QB, DB, σd)
                haskey(st.F, Fkey) || continue
                Fmat = st.F[Fkey]               # ⟨QB+1,DB+σd| f†_σ |QB,DB⟩
                pA = (QB + 1, DB + σd)
                for (sfrom, sto, amp) in moves
                    sfrom == sB || continue
                    haskey(rng, (pA, sto)) || continue
                    rA = rng[(pA, sto)]         # target segment |pA; sto⟩
                    T[rA, rB] .+= (coupling * amp * (-1)^QB) .* Fmat
                end
            end
        end
        H[nqn] = Diagonal(diagv) .+ T .+ transpose(T)
    end
    return U1U1Enlarged(H, seg)
end

"""
    block_levels(state, sym) -> Vector{Tuple{Float64,Int}}

The kept spectrum as `(energy, 2Sₙ)` pairs — the quantum-number-resolved levels
the thermodynamics/spectral layers consume. For `U1U1`, `2Sₙ = D` is the block
label.
"""
block_levels(st::U1U1State, ::U1U1) = [(e, D) for ((Q, D), ev) in st.E for e in ev]

function diagonalize_blocks(enl::U1U1Enlarged, ::U1U1)
    vals = Dict{NTuple{2,Int},Vector{Float64}}()
    vecs = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (qn, Hb) in enl.H
        F = eigen(Symmetric(Hb))
        vals[qn] = F.values
        vecs[qn] = Matrix(F.vectors)
    end
    return U1U1Diag(vals, vecs, enl.seg)
end

# ---- truncate + rebuild operators in the kept eigenbasis ----
function update_operators(diag::U1U1Diag, plan::Dict{NTuple{2,Int},Vector{Int}}, ::U1U1)
    Enew = Dict{NTuple{2,Int},Vector{Float64}}()
    Vk = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (qn, idx) in plan
        Enew[qn] = diag.vals[qn][idx]
        Vk[qn] = diag.vecs[qn][:, idx]
    end
    # Subtract the iteration ground energy (standard NRG): keeps the rescaled
    # spectrum O(1) under √Λ iteration and avoids catastrophic cancellation in the
    # low-energy levels on long chains. Eigenvectors / relative spectrum unchanged.
    g = minimum(minimum, values(Enew))
    for qn in keys(Enew)
        Enew[qn] = Enew[qn] .- g
    end
    Fnew = Dict{NTuple{3,Int},Matrix{Float64}}()
    for (qn, segs) in diag.seg
        haskey(Vk, qn) || continue
        Q, D = qn
        for (σd, moves) in _CREATE
            tgt = (Q + 1, D + σd)
            haskey(Vk, tgt) || continue
            # f†_new,σ in the product basis: |P;s⟩ → (−1)^{Q_P} |P; c†_σ s⟩ (block-diagonal in P)
            M = zeros(Float64, size(diag.vecs[tgt], 1), size(diag.vecs[qn], 1))
            tgtrng = Dict((p, s) => r for (p, s, r) in diag.seg[tgt])
            for (p, s, r) in segs
                for (sfrom, sto, amp) in moves
                    sfrom == s || continue
                    haskey(tgtrng, (p, sto)) || continue
                    rt = tgtrng[(p, sto)]
                    for (a, b) in zip(rt, r)        # same parent ⇒ identity in r
                        M[a, b] = amp * (-1)^(p[1])
                    end
                end
            end
            block = transpose(Vk[tgt]) * M * Vk[qn]
            iszero(block) || (Fnew[(Q, D, σd)] = block)
        end
    end
    return U1U1State(Enew, Fnew)
end
