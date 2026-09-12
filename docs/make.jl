using Documenter
using SDMXerWizard

makedocs(
    sitename = "SDMXerWizard.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://baffelan.github.io/SDMXerWizard.jl",
        assets = String[],
    ),
    modules = [SDMXerWizard],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => [
            "Data Sources" => "api/sources.md",
            "Data Profiling" => "api/profiling.md",
            "Mapping Inference" => "api/mapping.md",
            "Script Planning" => "api/scripts.md",
            "MCP Tools" => "api/mcp.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Baffelan/SDMXerWizard.jl.git",
    devbranch = "main",
)
