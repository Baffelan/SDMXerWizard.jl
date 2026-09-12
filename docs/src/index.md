# SDMXerWizard.jl

```@docs
SDMXerWizard.SDMXerWizard
```

## Features

- **Data profiling**: column-level type detection, pattern analysis, and summary statistics
- **Mapping inference**: fuzzy string matching, value pattern analysis against codelists, confidence scoring
- **Transformation planning**: ordered steps, templates, and the recodings a script still has to decide
- **Script execution and validation**: run a script written by the calling model and validate the result with SDMXer
- **MCP tool surface**: the whole flow exposed to Claude Code, Claude Desktop, or any other MCP client

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Baffelan/SDMXerWizard.jl")
```

SDMXerWizard depends on [SDMXer.jl](https://github.com/Baffelan/SDMXer.jl) for SDMx parsing and validation.

## Quick Example

```julia
using SDMXerWizard

session = Tools.Session()
schema = Tools.load_schema(session; endpoint = "SPC", dataflow_id = "DF_BP50")
source = Tools.load_source(session; path = "my_data.csv")
mappings = Tools.infer_mappings(session; source_id = source["source_id"], schema_id = schema["schema_id"])
```

See [MCP Tools](@ref) for the server and the full tool list.
