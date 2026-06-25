using WilsonNRG
using Documenter
using Downloads

assets_dir = joinpath(@__DIR__, "src", "assets")
mkpath(assets_dir)
favicon_path = joinpath(assets_dir, "favicon.ico")
logo_path = joinpath(assets_dir, "logo.png")

Downloads.download("https://github.com/sotashimozono.png", favicon_path)
Downloads.download("https://github.com/sotashimozono.png", logo_path)

makedocs(;
    sitename="WilsonNRG.jl",
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
    pages=["Home" => "index.md"],
)

deploydocs(;
    versions=["stable", "dev"],
    repo="github.com/sotashimozono/WilsonNRG.jl.git",
    devbranch="main",
    push_preview=true,
)
