# SDMXerWizard.jl

```@docs
SDMXerWizard.SDMXerWizard
```

## Features

- **Data profiling** -- column-level type detection, pattern analysis, and summary statistics
- **Mapping inference** -- fuzzy string matching, value pattern analysis, and optional LLM enhancement
- **Script generation** -- produce Tidier.jl or DataFrames.jl transformation scripts from mapping results
- **Workflow orchestration** -- end-to-end pipeline from source data to SDMX-validated output
- **Cross-dataflow analysis** -- LLM-assisted dataflow discovery, unit conversion, and join script generation
- **Prompt construction** -- build structured prompts for external LLM use without making API calls

## Installation

```julia
using Pkg
Pkg.add("SDMXerWizard")
```

SDMXerWizard depends on [SDMXer.jl](https://github.com/Baffelan/SDMXer.jl) for core SDMX parsing and validation.

## Quick Example

```julia
using SDMXer, SDMXerWizard

# Extract schema from an SDMX dataflow
schema = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50")

# Profile a source CSV
profile = profile_source_data("my_data.csv")

# Infer column mappings (heuristic, no LLM needed)
result = infer_mappings(profile, schema; method=:fuzzy)

# Generate a transformation script
gen = create_script_generator()
script = generate_transformation_script(gen, result, schema)
println(script.code)
```
