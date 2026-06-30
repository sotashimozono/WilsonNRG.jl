# ===========================================================================
#  Iterative-diagonalization engine for U(1)charge × SU(2)spin (`U1SU2`).
#
#  The non-abelian symmetry layer: spin *multiplets* (Q, S) are stored once (not the
#  2S+1 Sₙ members), so operators are spin tensors handled by Wigner–Eckart (reduced
#  matrix elements) and adding a site recouples the spins. Builds on the angular-momentum
#  foundation in su2.jl. Validated by reproducing the `U1U1` spectrum (2S+1)-expanded.
#
#  Conventions
#  -----------
#  * Multiplets labelled `(Q, S)`, Q = electron number, S = total spin (Rational).
#  * Electron site multiplets: |0⟩=(0,0), {|↑⟩,|↓⟩}=(1,½), |↑↓⟩=(2,0).
#  * `F[(Q,S,S′)]` = reduced matrix element ⟨(Q+1,S′)·‖f†_lastsite‖(Q,S)·⟩ (a matrix
#    over the kept multiplets of each block), the data propagated across iterations.
#  * HOPPING recoupling (scalar f†_N·f̃_{N+1}, derived + validated): the block element is
#    `coupling·(−1)^{s′+S_k+S+½}·√((2S_k′+1)(2s′+1))·{S_k′ s′ S; s S_k ½}·F·⟨s′‖f̃‖s⟩`.
#  * OPERATOR UPDATE (new-site f† reduced ME) uses the recoupling computed directly from
#    Clebsch–Gordan over the abstract multiplet (`_su2_recouple`, exact), × the fermion
#    sign `(−1)^{Q_parent}` — the Edmonds-7.1.8 6j written out numerically.
# ===========================================================================

using LinearAlgebra: Symmetric, eigen

# electron-site spin reduced MEs ⟨s′‖f†_site‖s⟩ and the conjugate annihilation ⟨b‖f̃‖a⟩
_su2_fdag(s, sp) = (s == 0 // 1 && sp == 1 // 2) ? 1.0 :
                   (s == 1 // 2 && sp == 0 // 1) ? -sqrt(2.0) : 0.0
_su2_ftil(sa, sb) = (-1.0)^(1 // 2 + sa - sb) * sqrt((2sa + 1) / (2sb + 1)) * _su2_fdag(sb, sa)
_su2_tri(a, b, c) = (abs(a - b) ≤ c ≤ a + b)

# site f† Sz-resolved ME ⟨s′ ms′|f†_μ|s ms⟩ (μ = ms′−ms), for the grounded recoupling
function _su2_site_me(s, ms, sp, msp, μ)
    msp == ms + μ || return 0.0
    (s == 0 // 1 && sp == 1 // 2) && return 1.0
    if s == 1 // 2 && sp == 0 // 1
        (μ == -1 // 2 && ms == 1 // 2) && return -1.0
        (μ == 1 // 2 && ms == -1 // 2) && return 1.0
    end
    return 0.0
end

# reduced ME ⟨(S_k,s′)S′‖f†_site‖(S_k,s)S⟩ (spin part) by explicit CG contraction over the
# spectator multiplet S_k — an exact Wigner 6-j evaluated numerically (no convention ambiguity).
function _su2_recouple(Sk, s, sp, S, Sp)
    for μ in (1 // 2, -1 // 2)
        Szp = S + μ
        abs(Szp) ≤ Sp || continue
        cgWE = clebsch_gordan(S, S, 1 // 2, μ, Sp, Szp)
        cgWE == 0 && continue
        me = 0.0
        for mk in (-Sk):Sk, ms in (-s):s, msp in (-sp):sp
            c1 = clebsch_gordan(Sk, mk, s, ms, S, S)
            c2 = clebsch_gordan(Sk, mk, sp, msp, Sp, Szp)
            (c1 == 0 || c2 == 0) && continue
            me += c1 * c2 * _su2_site_me(s, ms, sp, msp, μ)
        end
        return me / cgWE
    end
    return 0.0
end

# electron-site multiplets (Q, S)
const _SU2_SITE = ((0, 0 // 1), (1, 1 // 2), (2, 0 // 1))

"""
    U1SU2State

NRG state in the `(Q, S)` multiplet basis: kept eigen-energies per multiplet and the
reduced matrix elements `⟨(Q+1,S′)‖f†_σ‖(Q,S)⟩` of the last-site creation operator.
"""
struct U1SU2State
    E::Dict{Tuple{Int,Rational{Int}},Vector{Float64}}
    F::Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}   # (Q,S,S′) → ⟨(Q+1,S′)‖f†‖(Q,S)⟩
end

const _SU2Seg = Tuple{Int,Rational{Int},Int,Rational{Int},UnitRange{Int}}   # (Qk,Sk,q,s,range)

"Enlarged (pre-diagonalization) Hamiltonian blocks + product-basis segmentation, `(Q,S)`-keyed."
struct U1SU2Enlarged
    H::Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}
    seg::Dict{Tuple{Int,Rational{Int}},Vector{_SU2Seg}}
end

"Per-block eigendecomposition of an enlarged `U1SU2` Hamiltonian."
struct U1SU2Diag
    vals::Dict{Tuple{Int,Rational{Int}},Vector{Float64}}
    vecs::Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}
    seg::Dict{Tuple{Int,Rational{Int}},Vector{_SU2Seg}}
end

# ---- impurity (iteration −1): the 3-multiplet Anderson impurity ----
function impurity_init(m::AndersonModel, ::U1SU2, ::WilsonChain)
    E = Dict(
        (0, 0 // 1) => [0.0], (1, 1 // 2) => [m.εd], (2, 0 // 1) => [2 * m.εd + m.U]
    )
    F = Dict(
        (0, 0 // 1, 1 // 2) => fill(1.0, 1, 1),       # ⟨(1,½)‖d†‖(0,0)⟩ = 1
        (1, 1 // 2, 0 // 1) => fill(-sqrt(2.0), 1, 1), # ⟨(2,0)‖d†‖(1,½)⟩ = −√2
    )
    return U1SU2State(E, F)
end

# ---- attach one Wilson site (multiplet block recursion + 6j hopping) ----
function add_site(st::U1SU2State, ::U1SU2; coupling::Real, rescale::Real, onsite::Real=0.0)
    raw = Dict{Tuple{Int,Rational{Int}},Vector{Tuple{Int,Rational{Int},Int,Rational{Int}}}}()
    for ((Qk, Sk), Ev) in st.E, (q, s) in _SU2_SITE
        for twoS in Int(2 * abs(Sk - s)):2:Int(2 * (Sk + s))   # integer steps in S
            push!(get!(raw, (Qk + q, twoS // 2), valtype(raw)[]), (Qk, Sk, q, s))
        end
    end
    seg = Dict{Tuple{Int,Rational{Int}},Vector{_SU2Seg}}()
    for (nqn, lst) in raw
        sort!(lst)
        segs = _SU2Seg[]
        off = 0
        for (Qk, Sk, q, s) in lst
            d = length(st.E[(Qk, Sk)])
            push!(segs, (Qk, Sk, q, s, (off + 1):(off + d)))
            off += d
        end
        seg[nqn] = segs
    end
    H = Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}()
    for ((Q, S), segs) in seg
        dim = isempty(segs) ? 0 : last(last(segs)[5])
        Hb = zeros(Float64, dim, dim)
        for (Qk, Sk, q, s, r) in segs
            for (k, i) in enumerate(r)
                Hb[i, i] = rescale * st.E[(Qk, Sk)][k] + onsite * q
            end
        end
        # hopping: parent (Qk,Sk)→(Qk+1,Sk′) via F (f†_N), site s→s′ via f̃ (annihilation)
        for (Qk, Sk, q, s, r) in segs, (Qk2, Sk2, q2, s2, r2) in segs
            (Qk2 == Qk + 1 && q2 == q - 1) || continue
            (_su2_tri(Sk, 1 // 2, Sk2) && _su2_tri(s2, 1 // 2, s)) || continue
            haskey(st.F, (Qk, Sk, Sk2)) || continue
            Fm = st.F[(Qk, Sk, Sk2)]
            coef =
                coupling * (-1.0)^(s2 + Sk + S + 1 // 2) * sqrt((2Sk2 + 1) * (2s2 + 1)) *
                wigner6j(Sk2, s2, S, s, Sk, 1 // 2) * _su2_ftil(s, s2)
            for (a, ib) in enumerate(r2), (b, ik) in enumerate(r)
                Hb[ib, ik] += coef * Fm[a, b]
                Hb[ik, ib] += coef * Fm[a, b]                 # h.c. (real symmetric)
            end
        end
        H[(Q, S)] = Hb
    end
    return U1SU2Enlarged(H, seg)
end

function diagonalize_blocks(enl::U1SU2Enlarged, ::U1SU2)
    vals = Dict{Tuple{Int,Rational{Int}},Vector{Float64}}()
    vecs = Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}()
    for (qn, Hb) in enl.H
        F = eigen(Symmetric(Hb))
        vals[qn] = F.values
        vecs[qn] = Matrix(F.vectors)
    end
    return U1SU2Diag(vals, vecs, enl.seg)
end

"""
    block_levels(state, ::U1SU2) -> Vector{Tuple{Float64,Int}}

Kept multiplet spectrum as `(energy, 2S)` pairs — one entry per multiplet (the physical
degeneracy `2S+1` is supplied by [`multiplicity`](@ref)).
"""
block_levels(st::U1SU2State, ::U1SU2) = [(e, Int(2S)) for ((Q, S), ev) in st.E for e in ev]

# ---- truncate + rebuild the new-site f† reduced MEs in the kept eigenbasis ----
function update_operators(diag::U1SU2Diag, plan::Dict{K,Vector{Int}}, ::U1SU2) where {K}
    Enew = Dict{Tuple{Int,Rational{Int}},Vector{Float64}}()
    Vk = Dict{Tuple{Int,Rational{Int}},Matrix{Float64}}()
    for (qn, idx) in plan
        Enew[qn] = diag.vals[qn][idx]
        Vk[qn] = diag.vecs[qn][:, idx]
    end
    g = minimum(minimum, values(Enew))
    for qn in keys(Enew)
        Enew[qn] = Enew[qn] .- g
    end
    Fnew = Dict{Tuple{Int,Rational{Int},Rational{Int}},Matrix{Float64}}()
    for ((Q, S), segs) in diag.seg
        haskey(Vk, (Q, S)) || continue
        for S2 in (S - 1 // 2, S + 1 // 2)
            S2 ≥ 0 || continue
            tgt = (Q + 1, S2)
            haskey(Vk, tgt) || continue
            M = zeros(Float64, size(diag.vecs[tgt], 1), size(diag.vecs[(Q, S)], 1))
            tgtseg = Dict((Qk, Sk, q, s) => r for (Qk, Sk, q, s, r) in diag.seg[tgt])
            for (Qk, Sk, q, s, r) in segs
                for (qp, sp) in _SU2_SITE
                    qp == q + 1 || continue
                    _su2_fdag(s, sp) == 0.0 && continue
                    haskey(tgtseg, (Qk, Sk, qp, sp)) || continue
                    (_su2_tri(s, 1 // 2, sp) && _su2_tri(S, 1 // 2, S2)) || continue
                    coef = _su2_recouple(Sk, s, sp, S, S2) * (-1.0)^Qk
                    rt = tgtseg[(Qk, Sk, qp, sp)]
                    for (a, b) in zip(rt, r)
                        M[a, b] = coef             # diagonal on the spectator parent index
                    end
                end
            end
            blk = transpose(Vk[tgt]) * M * Vk[(Q, S)]
            iszero(blk) || (Fnew[(Q, S, S2)] = blk)
        end
    end
    return U1SU2State(Enew, Fnew)
end
