# Citations and the DOI cross-check

`WilsonNRG.jl` reproduces the published NRG literature: every concrete `AbstractImpurityModel`
/ `AbstractDiscretization` / `AbstractSymmetry` / `AbstractSpectralMethod` names the paper it
reproduces, and every faithfulness gate reproduces a value from that paper. A reproduction is
only as good as its citation — so the citation must be **precise** and **checked against the
paper itself** (no-cite-without-reproduction). This is the literature counterpart of "tests must
check an independent answer". Design adapted from [QAtlas.jl](https://github.com/sotashimozono/QAtlas.jl).

## 1. Cite a precise reference + equation, not "Author (Year)"

- The docstring / header of each method cites the **journal, volume, page, year** and the
  **specific equation / table** used, e.g. `Žitko & Pruschke, PRB 79, 085106 (2009), Eq. 36`.
- The full entry lives in `docs/reference.bib` with a **DOI** (`doi2bib`-quality: title,
  authors, journal, volume, year, **doi**). Add it there if missing.
- "Author (Year)" with no volume/page is not acceptable: the reader must be able to open the
  paper and land on the exact result.

A method with no traceable published reference does not go into `src/` as a reproduction.

## 2. Download the paper and cross-check — do not trust your own derivation alone

A self-derivation can be internally consistent and still wrong by a convention factor (spin
normalisation, a sign, per-site vs per-bond, a rescaled coupling, `V₀ = √(2DΓ/π)` vs a folded
`A_Λ`). A self-consistent check (two of your own routes) cannot catch this. Ground the convention
from the paper's own numbers:

1. **Fetch the source** with `doiget` (OA-first; falls back to metadata-only):

   ```bash
   doiget fetch 10.1103/PhysRevB.79.085106
   ```

   Open-access full texts are kept under `docs/refs/` (git-ignored). If paywalled with no OA
   copy, find the arXiv version or verify by an independent route, and say so in the PR.

2. **Read the published value in the paper's conventions.** PDFs are often protected — extract
   with `pypdf`:

   ```bash
   python3 -c "import pypdf; r=pypdf.PdfReader('paper.pdf'); [open(f'p{i}.txt','w').write(p.extract_text()) for i,p in enumerate(r.pages)]"
   ```

3. **Anchor the convention.** Match one clean, unambiguous published coefficient to the code.
   Examples that pinned WilsonNRG conventions: the Žitko–Pruschke fix `E₁(z)=(1−Λ^{−z})/lnΛ+1−z`
   (Eq. 36) makes `A_{f0}=ρ` exactly — reproducing that pinned the discretization; the free-fermion
   subset-sum spectrum pins the `√Λ` recursion and fermion signs. Guessing the convention (the
   earlier log-mean discretization) gave band-edge artefacts; the paper's Eq. 36 fixed it.

This is why the gates check **independent** published/closed-form answers, not tautological
self-consistency.

## 3. References live in `docs/reference.bib`, verified in CI

The same `reference.bib` that the docs bibliography renders is checked two ways (both read
`QATLAS_REFERENCES_BIB`, so they never disagree on which file is canonical):

1. **Stage 1 — Julia** (`test/core/test_references_bib.jl`): every paper cited in a `src/`
   docstring (by `volume, page`) resolves to a `reference.bib` entry, keys are unique, and every
   entry carries a well-formed DOI. Catches dangling / missing / **fabricated** references — it
   would fail on a non-existent id like the "PRB 57, 10287 (1998)" a manual audit once turned up
   (the real Bulla–Hewson–Pruschke paper is J. Phys.: Condens. Matter 10, 8365 (1998)).
2. **Stage 2 — CI** (`.github/workflows/VerifyReferences.yml`): every `reference.bib` DOI / arXiv
   id actually resolves upstream (Crossref / arXiv) via the `doiget verify` action, without
   downloading PDFs.

Keep keys stable; reuse an existing key rather than adding a near-duplicate. Never invent a
DOI — use a real, resolved entry or none at all.
