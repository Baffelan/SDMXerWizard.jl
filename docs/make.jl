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
            "Data Profiling" => "api/profiling.md",
            "Mapping Inference" => "api/mapping.md",
            "Script Generation" => "api/scripts.md",
            "Workflow" => "api/workflow.md",
            "Cross-Dataflow" => "api/crossdataflow.md",
            "LLM Integration" => "api/llm.md",
            "Prompts" => "api/prompts.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Baffelan/SDMXerWizard.jl.git",
    devbranch = "main",
)
