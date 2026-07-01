# test/core/test_references_bib.jl
#
# Stage 1 of the two-stage citation check (design: rules/citations.md, adapted from
# QAtlas.jl): every paper cited in a `src/` docstring resolves to an entry in
# docs/reference.bib — catches dangling / missing / fabricated references (e.g. the
# non-existent "PRB 57, 10287 (1998)" that a manual audit turned up), plus keeps the
# bibliography well-formed (unique keys, every entry DOI-bearing, DOIs well-formed).
#
#   Stage 2 (CI, doiget verify): every reference.bib DOI actually exists upstream —
#   see .github/workflows/VerifyReferences.yml. Both read QATLAS_REFERENCES_BIB, so
#   the two checks never disagree on which file is canonical.
#
# WilsonNRG cites in free text ("PRB 74, 245114 (2006)") rather than bibkeys, so the
# key-consistency check matches a cited (volume, first-page) against the bib's volume/
# pages fields — a (volume, page) pair pins a paper uniquely.

using WilsonNRG, Test

references_bib_path() =
    get(ENV, "QATLAS_REFERENCES_BIB", joinpath(pkgdir(WilsonNRG), "docs", "reference.bib"))

# (key, volume, first-page, doi) for each @entry in the .bib (volume precedes pages within an entry)
function bib_entries(path::AbstractString)
    entries = NamedTuple{(:key, :vol, :page, :doi),Tuple{String,Union{Int,Nothing},Union{Int,Nothing},String}}[]
    key = ""; vol = nothing; page = nothing; doi = ""
    flush!() = isempty(key) || push!(entries, (; key, vol, page, doi))
    for line in eachline(path)
        m = match(r"^@\w+\{\s*([^,\s]+)\s*,", line)
        if m !== nothing
            flush!(); key = String(m.captures[1]); vol = nothing; page = nothing; doi = ""
            continue
        end
        v = match(r"volume\s*=\s*\{(\d+)\}", line); v !== nothing && (vol = parse(Int, v.captures[1]))
        p = match(r"pages\s*=\s*\{0*(\d+)", line); p !== nothing && (page = parse(Int, p.captures[1]))
        d = match(r"doi\s*=\s*\{([^}]+)\}", line); d !== nothing && (doi = String(d.captures[1]))
    end
    flush!()
    return entries
end

# (volume, first-page) cited in any src/ docstring, keyed on a journal token so "Eq. (3)" etc. don't match
function src_citations()
    pat = r"(?:PRB|PRL|RMP|Rev\. ?Mod\. ?Phys\.|Phys\. ?Rev\. ?Lett\.|Phys\. ?Rev\. ?B|Phys\. ?Rev\.|J\. ?Phys\.:? ?Condens\.? ?Matter)\s+(\d+),?\s+0*(\d+)\s+\((\d{4})\)"
    srcdir = joinpath(pkgdir(WilsonNRG), "src")
    cited = Set{Tuple{Int,Int}}()
    for f in readdir(srcdir; join=true)
        endswith(f, ".jl") || continue
        for mt in eachmatch(pat, read(f, String))
            push!(cited, (parse(Int, mt.captures[1]), parse(Int, mt.captures[2])))
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
    @test length(unique(keys)) == length(keys)                       # no duplicate / near-dup keys
    @testset "every entry has a well-formed DOI" begin
        for e in entries
            @test !isempty(e.doi)                                    # doi2bib-quality: DOI present
            @test occursin(r"^10\.\d{4,9}/\S+$", e.doi)              # well-formed DOI
        end
    end

    # ---- every paper cited in src/ has a matching bib entry (vol, first-page) ----
    #      would have failed on the fabricated "PRB 57, 10287" and on Peters/Campo/Hofstetter
    #      before their entries were added.
    bibvp = Set((e.vol, e.page) for e in entries if e.vol !== nothing && e.page !== nothing)
    cited = src_citations()
    @test !isempty(cited)                                            # the scanner actually found citations
    missing_refs = filter(vp -> !(vp in bibvp), collect(cited))
    if !isempty(missing_refs)
        @info "src citations with no reference.bib entry (volume, page)" missing_refs
    end
    @test isempty(missing_refs)
end
