# Getting Started

## Two ways in

SDMXerWizard can be driven by a language model through MCP, or called from
Julia directly. Both paths use the same `Tools` functions and the same core.

## From Claude Code

Add a `.mcp.json` to the project you work in:

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

Then ask, in order: load the dataflow, load the file, infer mappings, look up
codes where the mapping is unsure, plan the transformation, write and run the
script until it validates. The [MCP Tools](@ref) page lists what each tool
returns.

## From Julia

### 1. Load a schema

```julia
using SDMXerWizard

session = Tools.Session()
schema = Tools.load_schema(session;
    url = "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
schema["required_columns"]
```

### 2. Load and profile a source file

```julia
source = Tools.load_source(session; path = "balance_of_payments.csv")
source["columns"]
```

### 3. Infer mappings

```julia
mappings = Tools.infer_mappings(session;
    source_id = source["source_id"], schema_id = schema["schema_id"], method = "advanced")
mappings["unmapped_required_columns"]
```

### 4. Plan the transformation

```julia
plan = Tools.transformation_plan(session;
    source_id = source["source_id"], schema_id = schema["schema_id"],
    mappings = Dict("REF_AREA" => "Country"))
plan["recodings"]
plan["contract"]
```

### 5. Run a script and validate

```julia
script = """
result = DataFrame(REF_AREA = source[!, "Country"], TIME_PERIOD = string.(source[!, "Year"]),
                   OBS_VALUE = Float64.(source[!, "Value"]))
"""
report = Tools.run_script(session;
    source_id = source["source_id"], schema_id = schema["schema_id"],
    script = script, output_path = "out.csv")
report["validation"]["compliance_status"]
```

## Core functions

The struct-returning core is still available for programmatic use:

```julia
schema = extract_dataflow_schema(url)
data = read_source_data("balance_of_payments.csv")
profile = profile_source_data(data, "balance_of_payments.csv")
result = infer_mappings(profile, schema; method=:fuzzy)
engine = create_inference_engine(fuzzy_threshold=0.7)
advanced = infer_advanced_mappings(engine, profile, schema, data)
steps = build_transformation_steps(advanced, profile, schema)
```
