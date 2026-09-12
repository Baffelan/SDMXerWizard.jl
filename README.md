# SDMXerWizard.jl

[![Build Status](https://github.com/Baffelan/SDMXerWizard.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Baffelan/SDMXerWizard.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/Baffelan/SDMXerWizard.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Baffelan/SDMXerWizard.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A library that a language model uses to bring source data into SDMx shape.
Built on [SDMXer.jl](https://github.com/Baffelan/SDMXer.jl), it profiles a CSV
or Excel file, infers how its columns map to an SDMx dataflow, plans the
transformation, runs the script the model writes, and validates the result
against the dataflow structure. The model orchestrates the steps and writes
the code; the library supplies the facts and checks the outcome.

## Requirements

- Julia 1.12 or higher
- SDMXer.jl (installed as a dependency)
- An MCP client such as Claude Code or Claude Desktop, for the tool surface

## Philosophy

- The user does not need to know the SDMx REST API. As much as possible the
  package works from the developer API link given by the .Stat Data Explorer.
- Generative models are given eyes and a checker rather than answers. The
  library never calls a model itself. It exposes deterministic tools, the
  model reasons over their output, and every script the model writes is run
  and validated by the library before anyone trusts it.
- Raw data stays close to home. The model sees a handful of sample values per
  column, and `load_source` can anonymise a file before anything is returned.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Baffelan/SDMXerWizard.jl")
```

## Using it from Claude Code

Register the server in a `.mcp.json` at the root of the project you work in,
adjusting the project path:

```json
{
  "mcpServers": {
    "sdmxer-wizard": {
      "command": "julia",
      "args": ["--project=/path/to/SDMXerWizard.jl", "--startup-file=no",
               "-e", "using SDMXerWizard; serve_mcp()"]
    }
  }
}
```

Claude Code then discovers seven tools:

| Tool | What it does |
| --- | --- |
| `load_schema` | Fetch a dataflow structure from a .Stat URL or a local SDMx-ML file; returns a `schema_id` and a summary |
| `lookup_codes` | List or search the codes behind a dimension or attribute |
| `load_source` | Read and profile a CSV or Excel file; returns a `source_id` and a compact column profile |
| `infer_mappings` | Rank source columns against target columns, with confidence and evidence |
| `transformation_plan` | Ordered steps, a template, the recodings still to decide, and the contract a script must follow |
| `run_script` | Evaluate the model's Julia script against the loaded source and validate the result |
| `validate_csv` | Validate an SDMx-CSV file produced elsewhere |

Handles such as `schema_1` and `source_1` refer to objects held in the server
session, so later calls never re-fetch a dataflow or pass large objects
through the model's context. See `demo/korea/walkthrough.md` for a scripted
session.

## Using it from Julia

The same functions work without MCP. Each takes a `Session` and returns a
`Dict` of plain values.

```julia
using SDMXerWizard

session = Tools.Session()
schema = Tools.load_schema(session; url = "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
source = Tools.load_source(session; path = "my_data.csv")
mappings = Tools.infer_mappings(session; source_id = source["source_id"], schema_id = schema["schema_id"])
plan = Tools.transformation_plan(session; source_id = source["source_id"], schema_id = schema["schema_id"],
                                 mappings = Dict("REF_AREA" => "Country"))
report = Tools.run_script(session; source_id = source["source_id"], schema_id = schema["schema_id"],
                          script = "result = select(source, ...)", output_path = "out.csv")
```

The core functions underneath return Julia structs and are exported as before:

```julia
using SDMXerWizard

schema = extract_dataflow_schema(url)
source_data = read_source_data("my_data.csv")
profile = profile_source_data(source_data, "my_data.csv")

engine = create_inference_engine(fuzzy_threshold=0.7)
mapping_result = infer_advanced_mappings(engine, profile, schema, source_data)
steps = build_transformation_steps(mapping_result, profile, schema)
```

### Anonymisation

`anonymize_source_data` replaces values with synthetic ones while keeping
structure, types and cardinality, so a file can be profiled and mapped
without its contents leaving the machine. From MCP, pass `anonymize: true`
to `load_source`.

```julia
anon = anonymize_source_data(source_data, profile; target_schema=schema)
cfg = AnonymizationConfig(:preserve_distribution, 1000, 50, true, true)
anon_q = anonymize_source_data(source_data, profile; target_schema=schema, config=cfg)
```

## Core API

| Function | Description |
| --- | --- |
| `read_source_data(file_path; kwargs...)` | Read CSV or Excel data |
| `profile_source_data(data, file_path)` | Profile data structure |
| `infer_mappings(source, schema; method)` | Mapping API with `:heuristic`, `:fuzzy` and `:advanced` methods |
| `create_inference_engine(kwargs...)` | Configure the mapping engine |
| `infer_advanced_mappings(engine, profile, schema, data)` | Run the engine |
| `validate_mapping_quality(result, schema)` | Coverage and warnings for a mapping result |
| `build_transformation_steps(mapping, profile, schema)` | Ordered transformation steps |
| `select_template(profile)` | Pick a script template for the source shape |
| `anonymize_source_data(data, profile; kwargs...)` | Synthetic stand-in for the data |

## Testing

```julia
using Pkg
Pkg.test("SDMXerWizard")
```

The tool tests run offline against a hand-built schema. One test starts the
MCP server in a second Julia process and drives it over stdio.

## Related packages

- [SDMXer.jl](https://github.com/Baffelan/SDMXer.jl): SDMx parsing, validation and joins
- [ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl): the MCP server implementation

## License

MIT
