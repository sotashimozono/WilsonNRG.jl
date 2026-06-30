# ===========================================================================
#  Kondo model under U(1)charge × U(1)spin — a SECOND impurity model on the SAME
#  generic engine. Only `impurity_init` differs from the Anderson case: the
#  impurity is a localized spin-½ (no charge) exchange-coupled to the first bath
#  orbital by `J S·s_{f₀}`. After this init the tracked operator is `f†_{0σ}` and
#  the recursion (add_site / diagonalize_blocks / update_operators) + the
#  thermodynamics/magnetics layers are reused unchanged — the point of a generic
#  NRG *scheme*.
#
#  Built programmatically (8-state spin⊗f₀, explicit operators → block-diagonalize
#  → rotate f†₀) rather than hand-derived, so the Clebsch-Gordan / fermion-sign
#  bookkeeping is generated, not transcribed. The impurity spin carries no charge
#  and commutes with the conduction fermions ⇒ no Jordan–Wigner sign on f₀ here.
# ===========================================================================

function impurity_init(m::KondoModel, ::U1U1, ::WilsonChain)
    J = m.J
    # product basis |imp ⊗ f₀⟩: imp a ∈ {1:⇑(2Sz=+1), 2:⇓(2Sz=-1)}, f₀ s ∈ 1:4 (|0⟩|↑⟩|↓⟩|↑↓⟩)
    twoSz_imp = (1, -1)
    idx(a, s) = (a - 1) * 4 + s                          # flat 1..8
    qd = NTuple{2,Int}[]
    for a in 1:2, s in 1:4
        push!(qd, (_LOC_Q[s], twoSz_imp[a] + _LOC_D[s]))  # (Q, 2Sz_total) of each product state
    end

    # ---- H = J [ Sz_imp·sz_{f₀} + ½(S⁺_imp s⁻_{f₀} + S⁻_imp s⁺_{f₀}) ] as an 8×8 ----
    H = zeros(Float64, 8, 8)
    szf0 = (0.0, 0.5, -0.5, 0.0)                          # sz of |0⟩,|↑⟩,|↓⟩,|↑↓⟩
    for a in 1:2, s in 1:4
        H[idx(a, s), idx(a, s)] += J * (twoSz_imp[a] / 2) * szf0[s]   # Sz·sz (diagonal)
    end
    # spin flips on the f₀ singly-occupied states: s⁻: |↑⟩(2)→|↓⟩(3); s⁺: |↓⟩(3)→|↑⟩(2)
    # S⁺_imp: ⇓(2)→⇑(1);  S⁻_imp: ⇑(1)→⇓(2)
    H[idx(1, 3), idx(2, 2)] += J / 2                      # S⁺_imp s⁻_{f₀}: |⇓↑⟩→|⇑↓⟩
    H[idx(2, 2), idx(1, 3)] += J / 2                      # h.c.
    H[idx(2, 2), idx(1, 3)] == H[idx(1, 3), idx(2, 2)] || error("Kondo H not symmetric")

    # ---- c†_{f₀σ} as 8×8 (imp identity ⊗ single-site creation; no JW sign on a spin) ----
    Cdag = Dict(σd => zeros(Float64, 8, 8) for (σd, _) in _CREATE)
    for (σd, moves) in _CREATE
        for a in 1:2, (sfrom, sto, amp) in moves
            Cdag[σd][idx(a, sto), idx(a, sfrom)] += amp
        end
    end

    # ---- block-diagonalize H, store kept energies + rotation per (Q,D) ----
    blocks = Dict{NTuple{2,Int},Vector{Int}}()           # (Q,D) → product indices
    for i in 1:8
        push!(get!(blocks, qd[i], Int[]), i)
    end
    E = Dict{NTuple{2,Int},Vector{Float64}}()
    V = Dict{NTuple{2,Int},Matrix{Float64}}()            # product-rows × eigen-cols
    for (b, is) in blocks
        F = eigen(Symmetric(H[is, is]))
        E[b] = F.values
        V[b] = Matrix(F.vectors)
    end

    # ---- f†_{0σ}[(Q,D,σd)] = V_tgt' · c†_σ[tgt,src] · V_src  (tgt = (Q+1, D+σd)) ----
    Fdag = Dict{NTuple{3,Int},Matrix{Float64}}()
    for (b, is) in blocks
        Q, D = b
        for (σd, _) in _CREATE
            tgt = (Q + 1, D + σd)
            haskey(blocks, tgt) || continue
            js = blocks[tgt]
            block = transpose(V[tgt]) * Cdag[σd][js, is] * V[b]
            iszero(block) || (Fdag[(Q, D, σd)] = block)
        end
    end
    return U1U1State(E, Fdag)
end

# The Kondo init already contains f₀ (via S·s), so nrg_solve's first attach is f₁.
bath_sites_in_init(::KondoModel) = 1
