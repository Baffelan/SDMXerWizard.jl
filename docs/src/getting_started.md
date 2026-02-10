# Getting Started

## LLM Provider Setup

SDMXerWizard uses [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) for LLM integration. Configure your preferred provider before using LLM-enhanced features:

```julia
using SDMXerWizard

# OpenAI (requires OPENAI_API_KEY environment variable)
setup_sdmx_llm(OPENAI; model="gpt-4o")

# Anthropic (requires ANTHROPIC_API_KEY)
setup_sdmx_llm(ANTHROPIC; model="claude-sonnet-4-20250514")

# Google Gemini (requires GOOGLE_API_KEY + GoogleGenAI.jl loaded)
using GoogleGenAI
setup_sdmx_llm(GOOGLE; model="gemini-2.0-flash")

# Local Ollama
setup_sdmx_llm(OLLAMA; model="llama3.1")
```

Core features (profiling, fuzzy mapping, script generation) work **without** any LLM provider.

## Basic Workflow

### 1. Extract an SDMX schema

```julia
using SDMXer

schema = extract_dataflow_schema(
    "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50"
)
```

### 2. Profile source data

```julia
using SDMXerWizard

profile = profile_source_data("balance_of_payments.csv")
print_source_profile(profile)
```

### 3. Infer column mappings

```julia
# Fuzzy matching (no LLM)
result = infer_mappings(profile, schema; method=:fuzzy)

# LLM-enhanced (requires provider setup)
result = infer_mappings(profile, schema; method=:llm)
```

### 4. Generate a transformation script

```julia
gen = create_script_generator()
script = generate_transformation_script(gen, result, schema)
println(script.code)
```

### 5. Full workflow (all steps in one call)

```julia
config = WorkflowConfig(
    source_path="balance_of_payments.csv",
    dataflow_url="https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50",
)
workflow = create_workflow(config)
result = execute_workflow(workflow)
println(generate_workflow_report(result))
```

## Cross-Dataflow Joins

For joining data from multiple SDMX dataflows, see the [Cross-Dataflow](@ref) API reference and the [SDMXer.jl documentation](https://baffelan.github.io/SDMXer.jl/dev/api/joins/) on cross-dataflow joins.
