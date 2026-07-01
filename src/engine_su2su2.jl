# ===========================================================================
#  Iterative-diagonalization engine for SU(2)charge × SU(2)spin (`SU2SU2`) — the maximal
#  symmetry of the *symmetric* single-impurity Anderson model (εd = −U/2, symmetric band).
#
#  The doubly-non-abelian layer: (isospin I, spin S) *multiplets* are stored once (the
#  (2I+1)(2S+1) members are implied), so operators are double tensors handled by Wigner–Eckart
#  in BOTH sectors and adding a site recouples the isospin AND the spin. Builds on su2.jl and
#  mirrors engine_u1su2.jl. Validated by reproducing the `U1U1` spectrum (2I+1)(2S+1)-expanded.
#
#  Charge-SU(2) (isospin): I_z = (N̂ − N_orb)/2, and the raising operator is STAGGERED,
#  Q⁺ = Σ_n (−1)ⁿ c†_{n↑}c†_{n↓}, so that [Q⁺, H_hop] = 0. The staggering is a PHASE (not an I_z
#  flip): I_z is the physical charge on every site; only Q⁺ carries (−1)ⁿ, so the odd-site isospin
#  doublet is {|0⟩:I_z=−½, −|↑↓⟩:I_z=+½}. The impurity is site 0 (even).
#
#  Single-orbital multiplets: {|0⟩,|↑↓⟩}=(I=½,S=0), {|↑⟩,|↓⟩}=(I=0,S=½). The fermion is a Nambu
#  (½,½) DOUBLE tensor (creation AND annihilation unified into one isospin tensor) — the essential
#  difference from `U1SU2`, where the U(1) charge keeps c† and c separate.
#
#  Conventions
#  -----------
#  * Multiplets labelled `(I, S)`, both Rational; F[(I,S,I′,S′)] = ⟨(I′,S′)‖Φ_lastsite‖(I,S)⟩.
#  * HOPPING (analytic doubled-6j, both sectors the U1SU2 form): the block element is
#    coupling · [iso: (−1)^{i′+I_k+I+½}√((2I_k′+1)(2i′+1)){I_k′ i′ I; i I_k ½}·f̃_iso(i,i′;parity)]
#             · [spin: same with the S's and f̃] · F_parent.
#  * OPERATOR UPDATE (new-site Φ reduced ME) uses the recoupling computed GROUNDED from the
#    explicit c†↑ component recoupled over the abstract spectator (`_su2su2_recouple`), × the
#    fermion sign `(−1)^{2I_k}` (the parent fermion parity, constant within an (I,S) multiplet).
#  * Isospin reduced ME carries a μ_I conjugate phase (0→½ opposite sign vs spin) + the odd-site
#    staggering; the c†↑-only recoupling avoids the annihilation components' conjugate phase.
# ===========================================================================

using LinearAlgebra: Symmetric, eigen

# ---- isospin fermion reduced MEs: μ_I conjugate phase (0→½ opposite sign vs spin) + staggering ----
_su2_fdag_iso(i, ip, parity) =
    if (i == 0 // 1 && ip == 1 // 2)
        (isodd(parity) ? 1.0 : -1.0)
    elseif (i == 1 // 2 && ip == 0 // 1)
        -sqrt(2.0)
    else
        0.0
    end
function _su2_ftil_iso(ia, ib, parity)
    f = _su2_fdag_iso(ib, ia, parity)
    f == 0.0 && return 0.0
    return (-1.0)^Int(1 // 2 + ia - ib) * sqrt((2ia + 1) / (2ib + 1)) * f
end

# ---- explicit single-orbital ops + members for the grounded operator-update recoupling ----
# occupation basis |0⟩,|↑⟩,|↓⟩,|↑↓⟩; staggering = −1 on the odd-site I_z=+½ (|↑↓⟩) member.
const _SU2SU2_CDU = [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 -1 0]      # c†_↑
const _SU2SU2_CDD = [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]      # c†_↓
const _SU2SU2_MIDX = Dict(
    (1 // 2, 0 // 1, -1 // 2, 0 // 1) => 1,
    (1 // 2, 0 // 1, 1 // 2, 0 // 1) => 4,
    (0 // 1, 1 // 2, 0 // 1, 1 // 2) => 2,
    (0 // 1, 1 // 2, 0 // 1, -1 // 2) => 3,
)
function _su2su2_mvec(I, S, iz, sz, parity)
    haskey(_SU2SU2_MIDX, (I, S, iz, sz)) || return zeros(4)
    v = zeros(4)
    v[_SU2SU2_MIDX[(I, S, iz, sz)]] =
        (isodd(parity) && I == 1 // 2 && iz == 1 // 2) ? -1.0 : 1.0
    return v
end
# c†_↑ member matrix element ⟨(ip,sp;ipz,spz)|c†_↑|(i,s;iz,sz)⟩ (with staggering)
function _su2su2_cdu_me(i, s, iz, sz, ip, sp, ipz, spz, parity)
    return _su2su2_mvec(ip, sp, ipz, spz, parity)' *
           _SU2SU2_CDU *
           _su2su2_mvec(i, s, iz, sz, parity)
end

const _SU2SU2_SITE = [(1 // 2, 0 // 1), (0 // 1, 1 // 2)]

# grounded double recoupling ⟨(I_k,ip)Ip (S_k,sp)Sp‖Φ_site‖(I_k,i)I (S_k,s)S⟩: the c†↑ component
# (μ=+½ in both sectors), extracting the reduced ME from any valid nonzero-CG source weight —
# c†↑ alone reaches every Ip=I±½ / Sp=S±½ (via lower weights) and avoids the annihilation
# components' conjugate phase. Wigner–Eckart ⇒ weight-independent (consistency-asserted → NaN).
const _SU2SU2_RECOUPLE_CACHE = Dict{Any,Float64}()   # pure fn of the quantum numbers ⊕ site parity

function _su2su2_recouple(Ik, Sk, i, s, ip, sp, I, S, Ip, Sp, parity)
    return get!(
        _SU2SU2_RECOUPLE_CACHE, (Ik, Sk, i, s, ip, sp, I, S, Ip, Sp, isodd(parity))
    ) do
        return _su2su2_recouple_compute(Ik, Sk, i, s, ip, sp, I, S, Ip, Sp, parity)
    end
end
function _su2su2_recouple_compute(Ik, Sk, i, s, ip, sp, I, S, Ip, Sp, parity)
    result = nothing
    for twoMI in Int(2I):-2:Int(-2I), twoMS in Int(2S):-2:Int(-2S)
        MI = twoMI // 2
        MS = twoMS // 2
        MIp = MI + 1 // 2
        MSp = MS + 1 // 2
        (abs(MIp) ≤ Ip && abs(MSp) ≤ Sp) || continue
        cgWEi = clebsch_gordan(I, MI, 1 // 2, 1 // 2, Ip, MIp)
        cgWEi == 0 && continue
        cgWEs = clebsch_gordan(S, MS, 1 // 2, 1 // 2, Sp, MSp)
        cgWEs == 0 && continue
        me = 0.0
        for mIk in (-Ik):Ik, miz in (-i):i, mipz in (-ip):ip
            cI1 = clebsch_gordan(Ik, mIk, i, miz, I, MI)
            cI1 == 0 && continue
            cI2 = clebsch_gordan(Ik, mIk, ip, mipz, Ip, MIp)
            cI2 == 0 && continue
            for mSk in (-Sk):Sk, msz in (-s):s, mspz in (-sp):sp
                cS1 = clebsch_gordan(Sk, mSk, s, msz, S, MS)
                cS1 == 0 && continue
                cS2 = clebsch_gordan(Sk, mSk, sp, mspz, Sp, MSp)
                cS2 == 0 && continue
                me +=
                    cI1 *
                    cI2 *
                    cS1 *
                    cS2 *
                    _su2su2_cdu_me(i, s, miz, msz, ip, sp, mipz, mspz, parity)
            end
        end
        val = me / (cgWEi * cgWEs)
        if result === nothing
            (result = val)
        else
            (isapprox(result, val; atol=1.0e-9) || return NaN)
        end
    end
    return result === nothing ? 0.0 : result
end

"""
    SU2SU2State

NRG state in the `(I, S)` multiplet basis: kept eigen-energies per multiplet, the reduced matrix
elements `F[(I,S,I′,S′)] = ⟨(I′,S′)‖Φ‖(I,S)⟩` of the last-site fermion double tensor, and `p`, the
parity of the next site to attach (the isospin staggering alternates by site).
"""
struct SU2SU2State
    E::Dict{Tuple{Rational{Int},Rational{Int}},Vector{Float64}}
    F::Dict{NTuple{4,Rational{Int}},Matrix{Float64}}   # (I,S,I′,S′) → ⟨(I′,S′)‖Φ‖(I,S)⟩
    p::Int
end

const _SU2SU2Seg = Tuple{
    Rational{Int},Rational{Int},Rational{Int},Rational{Int},UnitRange{Int}
}

"Enlarged (pre-diagonalization) Hamiltonian blocks + product-basis segmentation, `(I,S)`-keyed."
struct SU2SU2Enlarged
    H::Dict{Tuple{Rational{Int},Rational{Int}},Matrix{Float64}}
    seg::Dict{Tuple{Rational{Int},Rational{Int}},Vector{_SU2SU2Seg}}
    p::Int
end

"Per-block eigendecomposition of an enlarged `SU2SU2` Hamiltonian."
struct SU2SU2Diag
    vals::Dict{Tuple{Rational{Int},Rational{Int}},Vector{Float64}}
    vecs::Dict{Tuple{Rational{Int},Rational{Int}},Matrix{Float64}}
    seg::Dict{Tuple{Rational{Int},Rational{Int}},Vector{_SU2SU2Seg}}
    p::Int
end

"`multiplicity(::SU2SU2, (I,S))` = (2I+1)(2S+1) — the physical states in an isospin-I spin-S multiplet."
multiplicity(::SU2SU2, qn) = Int((2 * qn[1] + 1) * (2 * qn[2] + 1))

# ---- impurity (site 0, even): the symmetric-point Anderson impurity ----
function impurity_init(m::AndersonModel, ::SU2SU2, ::WilsonChain)
    isapprox(m.εd, -m.U / 2; atol=1.0e-12) || throw(
        EngineUnimplemented(
            "SU2SU2 is the maximal symmetry of the SYMMETRIC Anderson model — it needs the " *
            "particle–hole symmetric point εd = −U/2 (got εd=$(m.εd), U=$(m.U)); use U1U1/U1SU2 off it",
        ),
    )
    # (½,0) = {|0⟩,|↑↓⟩} degenerate at E=0; (0,½) = {|↑⟩,|↓⟩} at E=εd.
    E = Dict((1 // 2, 0 // 1) => [0.0], (0 // 1, 1 // 2) => [m.εd])
    F = Dict{NTuple{4,Rational{Int}},Matrix{Float64}}()
    for (I, S) in _SU2SU2_SITE, (Ip, Sp) in _SU2SU2_SITE
        (_su2_tri(I, 1 // 2, Ip) && _su2_tri(S, 1 // 2, Sp)) || continue
        v = _su2_fdag_iso(I, Ip, 0) * _su2_fdag(S, Sp)     # impurity = even (parity 0)
        v == 0.0 && continue
        F[(I, S, Ip, Sp)] = fill(v, 1, 1)
    end
    return SU2SU2State(E, F, 1)                             # next site (f₀) has parity 1
end

# ---- attach one Wilson site (multiplet block recursion + doubled-6j hopping) ----
function add_site(
    st::SU2SU2State, ::SU2SU2; coupling::Real, rescale::Real, onsite::Real=0.0
)
    isapprox(onsite, 0.0; atol=1.0e-12) || throw(
        EngineUnimplemented(
            "SU2SU2 needs a symmetric band (onsite εₙ=0; got $onsite) for isospin symmetry",
        ),
    )
    par = st.p
    raw = Dict{Tuple{Rational{Int},Rational{Int}},Vector{NTuple{4,Rational{Int}}}}()
    for ((Ik, Sk), Ev) in st.E, (i, s) in _SU2SU2_SITE
        for twoI in Int(2 * abs(Ik - i)):2:Int(2 * (Ik + i)),
            twoS in Int(2 * abs(Sk - s)):2:Int(2 * (Sk + s))

            push!(
                get!(raw, (twoI // 2, twoS // 2), NTuple{4,Rational{Int}}[]), (Ik, Sk, i, s)
            )
        end
    end
    seg = Dict{Tuple{Rational{Int},Rational{Int}},Vector{_SU2SU2Seg}}()
    for (nqn, lst) in raw
        sort!(lst)
        segs = _SU2SU2Seg[]
        off = 0
        for (Ik, Sk, i, s) in lst
            d = length(st.E[(Ik, Sk)])
            push!(segs, (Ik, Sk, i, s, (off + 1):(off + d)))
            off += d
        end
        seg[nqn] = segs
    end
    H = Dict{Tuple{Rational{Int},Rational{Int}},Matrix{Float64}}()
    for ((I, S), segs) in seg
        dim = isempty(segs) ? 0 : last(last(segs)[5])
        Hb = zeros(Float64, dim, dim)
        for (Ik, Sk, i, s, r) in segs
            for (k, ii) in enumerate(r)
                Hb[ii, ii] = rescale * st.E[(Ik, Sk)][k]         # symmetric band ⇒ no onsite term
            end
        end
        # hopping: parent (Ik,Sk)→(Ik2,Sk2) via F, site (i,s)→(i2,s2), isospin & spin recoupling
        for (Ik, Sk, i, s, r) in segs, (Ik2, Sk2, i2, s2, r2) in segs
            (
                _su2_tri(Ik, 1 // 2, Ik2) &&
                _su2_tri(i2, 1 // 2, i) &&
                _su2_tri(Sk, 1 // 2, Sk2) &&
                _su2_tri(s2, 1 // 2, s)
            ) || continue
            haskey(st.F, (Ik, Sk, Ik2, Sk2)) || continue
            Fm = st.F[(Ik, Sk, Ik2, Sk2)]
            iso =
                (-1.0)^(i2 + Ik + I + 1 // 2) *
                sqrt((2Ik2 + 1) * (2i2 + 1)) *
                wigner6j(Ik2, i2, I, i, Ik, 1 // 2) *
                _su2_ftil_iso(i, i2, par)
            spn =
                (-1.0)^(s2 + Sk + S + 1 // 2) *
                sqrt((2Sk2 + 1) * (2s2 + 1)) *
                wigner6j(Sk2, s2, S, s, Sk, 1 // 2) *
                _su2_ftil(s, s2)
            coef = coupling * iso * spn
            (coef == 0.0 || iszero(Fm)) && continue
            # the triangle condition visits BOTH (Ik→Ik2) and (Ik2→Ik) orderings; fill one triangle
            # per ordering and symmetrize once (h.c.) — adding both here would double-count.
            for (a, ib) in enumerate(r2), (b, ik) in enumerate(r)
                Hb[ib, ik] += coef * Fm[a, b]
            end
        end
        H[(I, S)] = Matrix(Symmetric(Hb))
    end
    return SU2SU2Enlarged(H, seg, par)
end

function diagonalize_blocks(enl::SU2SU2Enlarged, ::SU2SU2)
    vals = Dict{Tuple{Rational{Int},Rational{Int}},Vector{Float64}}()
    vecs = Dict{Tuple{Rational{Int},Rational{Int}},Matrix{Float64}}()
    for (qn, Hb) in enl.H
        F = eigen(Symmetric(Hb))
        vals[qn] = F.values
        vecs[qn] = Matrix(F.vectors)
    end
    return SU2SU2Diag(vals, vecs, enl.seg, enl.p)
end

"""
    block_levels(state, ::SU2SU2) -> Vector{Tuple{Float64,Int}}

Kept multiplet spectrum as `(energy, (2I+1)(2S+1))` pairs — one entry per `(I,S)` multiplet, the
second element the full physical degeneracy (unlike `U1SU2`'s `2S`; SU2SU2 is doubly non-abelian).
"""
function block_levels(st::SU2SU2State, ::SU2SU2)
    return [(e, multiplicity(SU2SU2(), (I, S))) for ((I, S), ev) in st.E for e in ev]
end

# ---- truncate + rebuild the new-site Φ reduced MEs in the kept eigenbasis ----
function update_operators(diag::SU2SU2Diag, plan::Dict{K,Vector{Int}}, ::SU2SU2) where {K}
    Enew = Dict{Tuple{Rational{Int},Rational{Int}},Vector{Float64}}()
    Vk = Dict{Tuple{Rational{Int},Rational{Int}},Matrix{Float64}}()
    for (qn, idx) in plan
        Enew[qn] = diag.vals[qn][idx]
        Vk[qn] = diag.vecs[qn][:, idx]
    end
    isempty(Enew) && throw(
        ArgumentError("truncation kept no states; loosen EnergyCut or increase KeepN")
    )
    g = minimum(minimum, values(Enew))
    for qn in keys(Enew)
        Enew[qn] = Enew[qn] .- g
    end
    par = diag.p
    F = Dict{NTuple{4,Rational{Int}},Matrix{Float64}}()
    for ((I, S), segs) in diag.seg
        haskey(Vk, (I, S)) || continue
        for Ip in (I - 1 // 2, I + 1 // 2), Sp in (S - 1 // 2, S + 1 // 2)
            (Ip ≥ 0 && Sp ≥ 0) || continue
            haskey(Vk, (Ip, Sp)) || continue
            M = zeros(Float64, size(diag.vecs[(Ip, Sp)], 1), size(diag.vecs[(I, S)], 1))
            tgtseg = Dict((Ik, Sk, i, s) => r for (Ik, Sk, i, s, r) in diag.seg[(Ip, Sp)])
            for (Ik, Sk, i, s, r) in segs
                for (ii, ss) in _SU2SU2_SITE
                    (_su2_tri(i, 1 // 2, ii) && _su2_tri(s, 1 // 2, ss)) || continue
                    haskey(tgtseg, (Ik, Sk, ii, ss)) || continue
                    coef =
                        _su2su2_recouple(Ik, Sk, i, s, ii, ss, I, S, Ip, Sp, par) *
                        (-1.0)^Int(2Ik)
                    # fail closed: a NaN means _su2su2_recouple hit a Wigner-Eckart inconsistency —
                    # surface it rather than silently poisoning the Hamiltonian (NaN≠0 slips the guard).
                    isnan(coef) && error(
                        "SU2SU2 operator-update recoupling produced NaN (Wigner-Eckart violation) " *
                        "at (I,S)=($I,$S)→($Ip,$Sp), spectator (Ik,Sk)=($Ik,$Sk), site ($i,$s)→($ii,$ss)",
                    )
                    coef == 0.0 && continue
                    rt = tgtseg[(Ik, Sk, ii, ss)]
                    for (a, b) in zip(rt, r)
                        M[a, b] = coef                            # diagonal on the spectator parent index
                    end
                end
            end
            blk = transpose(Vk[(Ip, Sp)]) * M * Vk[(I, S)]
            iszero(blk) || (F[(I, S, Ip, Sp)] = blk)
        end
    end
    return SU2SU2State(Enew, F, par + 1)                          # next site: parity + 1
end
