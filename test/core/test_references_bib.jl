# test/core/test_references_bib.jl
#
# Stage 1 of the two-stage citation check (design: rules/citations.md, adapted from
# QAtlas.jl): every reference cited in a `src/` docstring via `[key](@cite)` resolves to
# an entry in docs/reference.bib — catches a dangling / mistyped / fabricated citation key
# offline, on every PR — plus keeps the bibliography well-formed (unique keys, every entry
# DOI-bearing, DOIs well-formed).
#
#   Stage 2 (CI, doiget verify): every reference.bib DOI actually exists upstream —
#   see .github/workflows/VerifyReferences.yml. Both read QATLAS_REFERENCES_BIB, so the
#   two checks never disagree on which file is canonical.
#
# Convention (rules/citations.md): src docstrings cite with `[<bibkey>](@cite)`, where
# <bibkey> is the doiget-generated key (e.g. `doi_10.1103_PhysRevB.79.085106`) that
# DocumenterCitations resolves to the References page. The bibkey IS doiget's citation
# key, so a reference added with `doiget cite` drops in without renaming.

using WilsonNRG, Test

function references_bib_path()
    return get(
        ENV, "QATLAS_REFERENCES_BIB", joinpath(pkgdir(WilsonNRG), "docs", "reference.bib")
    )
end

# (key, doi) for each @entry in the .bib.
function bib_entries(path::AbstractString)
    entries = NamedTuple{(:key, :doi),Tuple{String,String}}[]
    key = ""
    doi = ""
    flush!() = isempty(key) || push!(entries, (; key, doi))
    for line in eachline(path)
        m = match(r"^@\w+\{\s*([^,\s]+)\s*,", line)
        if m !== nothing
            flush!()
            key = String(m.captures[1])
            doi = ""
            continue
        end
        d = match(r"doi\s*=\s*\{([^}]+)\}", line)
        d !== nothing && (doi = String(d.captures[1]))
    end
    flush!()
    return entries
end

# bibkeys cited in any src/ file via [key](@cite) / [key](@citet) / [key](@citep).
function src_citations()
    pat = r"\[([^\]]+)\]\(@cite\w*\)"
    srcdir = joinpath(pkgdir(WilsonNRG), "src")
    cited = Set{String}()
    for f in readdir(srcdir; join=true)
        endswith(f, ".jl") || continue
        for mt in eachmatch(pat, read(f, String))
            push!(cited, String(mt.captures[1]))
        end
    end
    return cited
end

@testset "reference.bib integrity (citation check, stage 1)" begin
    path = references_bib_path()
    @test isfile(path)
    entries = bib_entries(path)

    # ---- bibliography is well-formed: non-trivial, unique keys, every entry DOI-bearing ----
    @test length(entries) ≥ 11
    keys = [e.key for e in entries]
    @test length(unique(keys)) == length(keys)                       # no duplicate keys
    @testset "every entry has a well-formed DOI" begin
        for e in entries
            @test !isempty(e.doi)                                    # doi2bib-quality: DOI present
            @test occursin(r"^10\.\d{4,9}/\S+$", e.doi)              # well-formed DOI
        end
    end

    # ---- every [key](@cite) in src/ resolves to a bib entry ----
    #      catches a dangling / mistyped / fabricated citation key (the @cite counterpart of
    #      the fabricated "PRB 57, 10287 (1998)" a manual audit once turned up).
    bibkeys = Set(e.key for e in entries)
    cited = src_citations()
    @test !isempty(cited)                                            # the scanner actually found @cite keys
    missing_refs = filter(k -> !(k in bibkeys), collect(cited))
    if !isempty(missing_refs)
        @info "src [key](@cite) with no reference.bib entry" missing_refs
    end
    @test isempty(missing_refs)
end
