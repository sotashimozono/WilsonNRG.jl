using WilsonNRG
using Documenter
using DocumenterCitations
using Downloads

assets_dir = joinpath(@__DIR__, "src", "assets")
mkpath(assets_dir)
favicon_path = joinpath(assets_dir, "favicon.ico")
logo_path = joinpath(assets_dir, "logo.png")

Downloads.download("https://github.com/sotashimozono.png", favicon_path)
Downloads.download("https://github.com/sotashimozono.png", logo_path)

bib = CitationBibliography(joinpath(@__DIR__, "reference.bib"); style=:numeric)

makedocs(;
    sitename="WilsonNRG.jl",
    plugins=[bib],
    format=Documenter.HTML(;
        canonical="https://codes.sota-shimozono.com/WilsonNRG.jl/stable/",
        prettyurls=get(ENV, "CI", "false") == "true",
        mathengine=MathJax3(
            Dict(
                :tex => Dict(
                    :inlineMath => [["\$", "\$"], ["\\(", "\\)"]],
                    :tags => "ams",
                    :packages => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
        assets=["assets/favicon.ico", "assets/custom.css"],
    ),
    modules=[WilsonNRG],
    pages=["Home" => "index.md", "References" => "references.md"],
)

deploydocs(;
    versions=["stable", "dev"],
    repo="github.com/sotashimozono/WilsonNRG.jl.git",
    devbranch="main",
    push_preview=true,
)
