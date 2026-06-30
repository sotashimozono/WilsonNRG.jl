# ===========================================================================
#  Foundation for the U(1)charge × SU(2)spin (`U1SU2`) symmetry layer.
#
#  Non-abelian NRG stores spin *multiplets* (Q, S) once (not the 2S+1 Sₙ members),
#  so operators are spin tensors handled by Wigner–Eckart (reduced matrix elements ×
#  Clebsch–Gordan) and adding a site recouples three spins via a 6j symbol. This
#  file provides the angular-momentum coefficients (verified against known values in
#  test/gates/test_su2_coeffs.jl) and the local electron-site multiplet structure.
#
#  STATUS: foundation only. The `impurity_init`/`add_site`/… methods for `U1SU2`
#  (the 6j recoupling of the ξ-hopping, validated by reproducing the `U1U1` spectrum
#  (2S+1)-expanded) are the next step; until then `U1SU2` raises `EngineUnimplemented`.
# ===========================================================================

# n! as Float64 (argument is integer-valued; good to 170! without overflow).
_facf(n) = prod(1.0:float(n))

_triangle(a, b, c) = (a + b ≥ c) && (a + c ≥ b) && (b + c ≥ a) && isinteger(a + b + c)

function _Δcoef(a, b, c)
    _triangle(a, b, c) || return 0.0
    return sqrt(
        _facf(a + b - c) * _facf(a - b + c) * _facf(-a + b + c) / _facf(a + b + c + 1)
    )
end

"""
    wigner3j(j1, j2, j3, m1, m2, m3) -> Float64

Wigner 3-j symbol (Racah formula). Arguments are multiples of ½.
"""
function wigner3j(j1, j2, j3, m1, m2, m3)
    (
        m1 + m2 + m3 == 0 &&
        _triangle(j1, j2, j3) &&
        abs(m1) ≤ j1 &&
        abs(m2) ≤ j2 &&
        abs(m3) ≤ j3 &&
        isinteger(j1 + m1) &&
        isinteger(j2 + m2) &&
        isinteger(j3 + m3)
    ) || return 0.0
    s = 0.0
    for t in 0:30
        d = (j3 - j2 + t + m1, j3 - j1 + t - m2, j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2)
        all(x -> x ≥ 0 && isinteger(x), d) || continue
        s +=
            (-1)^t /
            (_facf(t) * _facf(d[1]) * _facf(d[2]) * _facf(d[3]) * _facf(d[4]) * _facf(d[5]))
    end
    return (-1)^(j1 - j2 - m3) *
           _Δcoef(j1, j2, j3) *
           sqrt(
               _facf(j1 + m1) *
               _facf(j1 - m1) *
               _facf(j2 + m2) *
               _facf(j2 - m2) *
               _facf(j3 + m3) *
               _facf(j3 - m3),
           ) *
           s
end

"""
    clebsch_gordan(j1, m1, j2, m2, J, M) -> Float64

Clebsch–Gordan coefficient ⟨j1 m1; j2 m2 | J M⟩ (via the 3-j symbol).
"""
function clebsch_gordan(j1, m1, j2, m2, J, M)
    (-1)^(j1 - j2 + M) * sqrt(2J + 1) * wigner3j(j1, j2, J, m1, m2, -M)
end

"""
    wigner6j(a, b, c, d, e, f) -> Float64

Wigner 6-j symbol `{a b c; d e f}` (Racah formula) — the recoupling coefficient
that appears when adding a Wilson site to the kept multiplets.
"""
function wigner6j(a, b, c, d, e, f)
    (
        _triangle(a, b, c) && _triangle(c, d, e) && _triangle(a, e, f) && _triangle(b, d, f)
    ) || return 0.0
    s = 0.0
    for t in 0:30
        u = (
            t - a - b - c,
            t - c - d - e,
            t - a - e - f,
            t - b - d - f,
            a + b + d + e - t,
            a + c + d + f - t,
            b + c + e + f - t,
        )
        all(x -> x ≥ 0 && isinteger(x), u) || continue
        s +=
            (-1)^t * _facf(t + 1) / (
                _facf(u[1]) *
                _facf(u[2]) *
                _facf(u[3]) *
                _facf(u[4]) *
                _facf(u[5]) *
                _facf(u[6]) *
                _facf(u[7])
            )
    end
    return _Δcoef(a, b, c) * _Δcoef(c, d, e) * _Δcoef(a, e, f) * _Δcoef(b, d, f) * s
end

# Electron-site SU(2) multiplets (charge Q, spin S): |0⟩=(0,0), {|↑⟩,|↓⟩}=(1,½), |↑↓⟩=(2,0).
const _ELECTRON_MULTIPLETS = ((0, 0 // 1), (1, 1 // 2), (2, 0 // 1))

"`multiplicity(::U1SU2, (Q,S))` = 2S+1 — a spin-S multiplet is (2S+1) physical states."
multiplicity(::U1SU2, qn) = Int(2 * qn[2] + 1)
