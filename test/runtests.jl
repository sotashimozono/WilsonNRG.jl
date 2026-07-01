ENV["GKSwstype"] = "100"

using WilsonNRG
using Test, Aqua

include(joinpath(@__DIR__, "ci", "universe.jl"))     # ALL_TEST_FILES, test_file_key (WHAT to run)

# ── Test selection: FILES > ALL ──────────────────────────────────────
#   WILSONNRG_TEST_FILES="gates/test_a.jl,gates/test_b.jl" — explicit list emitted by the LPT shard
#       planner (test/ci/plan_shards.jl).  MUST be a subset of the canonical universe.
#   neither  — run everything (local `Pkg.test()` and the round-robin fallback).
# Aqua (whole-package QA) runs in exactly one selection: FILES → WILSONNRG_RUN_AQUA=1 ; ALL → yes.
const _test_files = get(ENV, "WILSONNRG_TEST_FILES", "")
const _selected, _mode, _run_aqua = if !isempty(_test_files)
    want = [strip(x) for x in split(_test_files, ",") if !isempty(strip(x))]
    idx = Dict(test_file_key(d, f) => (d, f) for (d, f) in ALL_TEST_FILES)
    sel = Tuple{String,String}[]
    unknown = String[]
    for w in want
        haskey(idx, w) ? push!(sel, idx[w]) : push!(unknown, String(w))
    end
    isempty(unknown) || error(
        "WILSONNRG_TEST_FILES lists files outside the canonical universe (planner must only emit " *
        "globbed files): $(unknown)",
    )
    (sel, "FILES (n=$(length(sel)))", get(ENV, "WILSONNRG_RUN_AQUA", "0") == "1")
else
    (ALL_TEST_FILES, "ALL", true)
end
println(
    "Test selection: $(_mode) → $(length(_selected))/$(length(ALL_TEST_FILES)) files; aqua=$(_run_aqua)",
)

const FIG_BASE = joinpath(pkgdir(WilsonNRG), "docs", "src", "assets")

# Per-file wall-time, captured for the timing plane (HOW-to-split the next run).
const _TIMINGS = Dict{String,Float64}()

@testset "tests" begin
    if _run_aqua
        @testset "Aqua tests" begin
            _TIMINGS["__aqua__"] = @elapsed Aqua.test_all(WilsonNRG)
            println("  Aqua: $(round(_TIMINGS["__aqua__"]; digits=2)) s")
        end
    end
    @time for (d, f) in _selected
        key = test_file_key(d, f)
        @testset "$(key)" begin
            println("  Including test/$(key)")
            _TIMINGS[key] = @elapsed include(joinpath(@__DIR__, d, f))
            println("  $(key): $(round(_TIMINGS[key]; digits=2)) s")
        end
    end
end

# Emit per-shard timing as TSV (key<TAB>seconds).  Gated by WILSONNRG_EMIT so only push:main CI
# writes; PR/local runs never persist.  The record-timings job merges these into the `ci-timings`
# orphan branch; the planner LPT-bin-packs the next run from it (round-robin until then).
if get(ENV, "WILSONNRG_EMIT", "0") == "1"
    outdir = get(ENV, "WILSONNRG_CIOUT_DIR", joinpath(@__DIR__, ".ci-out"))
    mkpath(outdir)
    sid = isempty(_test_files) ? "all" : string(hash(_test_files); base=16)
    tf = joinpath(outdir, "timings-$(sid).tsv")
    open(tf, "w") do io
        for (k, v) in sort!(collect(_TIMINGS); by=first)
            println(io, k, '\t', round(v; digits=4))
        end
    end
    println("Emitted timing TSV -> ", abspath(tf), " (", length(_TIMINGS), " entries)")
end
