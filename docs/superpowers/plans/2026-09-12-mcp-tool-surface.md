# MCP Tool Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SDMXerWizard into a deterministic library exposed to a language model through MCP tools, with the model writing the transformation script and the library validating the result.

**Architecture:** Three layers. The trimmed core keeps the deterministic code (sources, profiling, anonymisation, mapping inference, script templates and step builder). `SDMXerWizard.Tools` adds a `Session` with handle-keyed schemas and sources plus seven functions returning JSON-friendly `Dict`s. `src/MCPServer.jl` wraps those functions as `MCPTool`s and starts a stdio server whose protocol channel is a duplicate of the original stdout, with stdout itself redirected to stderr.

**Tech Stack:** Julia 1.12, SDMXer.jl, DataFrames, CSV, JSON3, ModelContextProtocol.jl 0.7.

**Spec:** `docs/superpowers/specs/2026-09-12-mcp-tool-surface-design.md`

## Global Constraints

- Branch `feature/mcp-tools`; no backward compatibility with 0.1.x.
- `julia = "1.12"` in compat; version `0.2.0`.
- No PromptingTools, OpenAI, GoogleGenAI or Llama anywhere in the package.
- Tool results contain only strings, numbers, booleans, `nothing`, vectors and dicts.
- Tools never write to stdout; the stdio protocol channel must stay clean.
- Julia code uses string concatenation, never interpolation.
- Prose (docs, README, commit messages) uses no em-dashes and no "not A, but B" constructions.
- Every commit message ends with the attribution lines given in the session.

## File map

| File | Responsibility |
|---|---|
| `Project.toml` | deps, compat, version |
| `src/SDMXerWizard.jl` | module header, includes, exports |
| `src/SDMXScriptGeneration.jl` | templates, `TransformationStep`, step builder, loading code (LLM path removed) |
| `src/SDMXMappingInference.jl` | mapping engine (LLM method removed; codelist preload honoured) |
| `src/Tools.jl` (new) | `Session`, `ToolError`, `run_tool`, seven tool functions, `jsonable` |
| `src/MCPServer.jl` (new) | `TOOL_SPECS`, `mcp_tools`, `serve_mcp` |
| `test/fixtures.jl` (new) | offline schema and codelists shared by tests |
| `test/test_tools.jl` (new) | tests for Tools and MCP wiring |
| `test/runtests.jl`, `test/simple_test.jl`, `test/test_korea_dryrun.jl` | trimmed |
| `.mcp.json` (new) | Claude Code registration |
| `demo/korea/inbound_visitors.csv`, `demo/korea/walkthrough.md` (new) | demo material |
| `docs/src/mcp.md` (new), `docs/make.jl`, `README.md` | documentation |

Deleted: `src/SDMXPromptingIntegration.jl`, `src/SDMXPromptGeneration.jl`, `src/SDMXCrossDataflowLLM.jl`, `src/SDMXJoinPrompts.jl`, `src/SDMXLLMPipelineOps.jl`, `src/SDMXEnhancedTransformation.jl`, `src/SDMXWorkflow.jl`, `src/SDMXJoinWorkflow.jl` (calls the LLM join script generator), `src/SDMXMetadataContext.jl` (exists only to build LLM prompt context and calls an undefined `analyze_excel_structure`), `ext/SDMXerWizardOpenAIExt.jl`, `ext/SDMXerWizardGoogleGenAIExt.jl`, `test/test_cross_dataflow.jl`, `docs/src/api/llm.md`, `docs/src/api/prompts.md`, `docs/src/api/workflow.md`, `docs/src/api/crossdataflow.md`.

---

### Task 1: Strip the internal LLM code

**Files:**
- Modify: `Project.toml`
- Modify: `src/SDMXerWizard.jl`
- Modify: `src/SDMXScriptGeneration.jl`
- Modify: `src/SDMXMappingInference.jl`
- Delete: the files listed under "Deleted" above (source, ext, test, docs)
- Modify: `test/runtests.jl`, `test/simple_test.jl`, `test/Project.toml`, `docs/make.jl`

**Interfaces:**
- Produces: `select_template(source_profile::SourceDataProfile; template_name::String="") -> ScriptTemplate`, `default_templates() -> Dict{String,ScriptTemplate}`, `SDMXerWizard.DEFAULT_TEMPLATE_NAME`. Everything else in the kept files keeps its current name and signature.

- [ ] **Step 1: Delete files**

```bash
git rm -q src/SDMXPromptingIntegration.jl src/SDMXPromptGeneration.jl src/SDMXCrossDataflowLLM.jl \
  src/SDMXJoinPrompts.jl src/SDMXLLMPipelineOps.jl src/SDMXEnhancedTransformation.jl \
  src/SDMXWorkflow.jl src/SDMXJoinWorkflow.jl src/SDMXMetadataContext.jl \
  ext/SDMXerWizardOpenAIExt.jl ext/SDMXerWizardGoogleGenAIExt.jl test/test_cross_dataflow.jl \
  docs/src/api/llm.md docs/src/api/prompts.md docs/src/api/workflow.md docs/src/api/crossdataflow.md
```

- [ ] **Step 2: Project.toml**

Remove `PromptingTools` from `[deps]`; remove `GoogleGenAI`, `Llama`, `OpenAI` from `[weakdeps]`, `[extensions]` and `[compat]`; remove the `PromptingTools` compat line. Add `ModelContextProtocol = "c58f755f-f2a7-4f48-bf29-4e9659b78499"` to `[deps]` and `ModelContextProtocol = "0.7"` to compat. Set `version = "0.2.0"` and `julia = "1.12"`. In `test/Project.toml` drop `PromptingTools` if present.

- [ ] **Step 3: Trim `src/SDMXScriptGeneration.jl`**

Delete `ScriptGenerator` (struct, docstring, callable methods), `create_script_generator`, `load_default_templates!`, `generate_transformation_script`, `generate_transformation_script_text`, `create_comprehensive_script_prompt`, `create_script_guidance`, `post_process_generated_code`, and every `ExcelStructureAnalysis` argument. Replace `select_template` with:

```julia
const DEFAULT_TEMPLATE_NAME = "standard_transformation"

"""
    default_templates() -> Dict{String, ScriptTemplate}

The four built-in script templates keyed by template name.
"""
function default_templates()
    return Dict{String, ScriptTemplate}(
        "standard_transformation" => create_standard_template(),
        "pivot_transformation" => create_pivot_template(),
        "excel_multi_sheet" => create_excel_template(),
        "simple_csv" => create_csv_template()
    )
end

"""
    select_template(source_profile::SourceDataProfile; template_name::String="") -> ScriptTemplate

Pick the built-in template that fits the source profile, or the named one.
"""
function select_template(source_profile::SourceDataProfile; template_name::String="")
    templates = default_templates()
    if !isempty(template_name) && haskey(templates, template_name)
        return templates[template_name]
    end
    if length(source_profile.suggested_time_columns) > 0 && source_profile.column_count > 5
        return templates["pivot_transformation"]
    end
    if source_profile.file_type == "csv" && source_profile.column_count <= 8
        return templates["simple_csv"]
    end
    return templates[DEFAULT_TEMPLATE_NAME]
end
```

Keep `GeneratedScript`, `validate_generated_script`, `preview_script_output`, `build_transformation_steps`, `get_loading_code`, `generate_transformation_logic`, `create_validation_logic`, `calculate_script_complexity` and the four `create_*_template` functions. Update the file docstring to say scripts are written by the calling model.

- [ ] **Step 4: Trim `src/SDMXMappingInference.jl`**

Delete the `:llm` branch of `infer_mappings` and its `llm_provider`/`llm_model` keyword arguments, `_parse_llm_mapping_text`, `_format_llm_mapping_result`, and any docstring lines mentioning LLM methods. In `load_codelist_data!`, after computing `codelist_refs`, add:

```julia
    if all(ref -> haskey(engine.codelists_data, ref), codelist_refs)
        return
    end
```

so a preloaded engine never fetches.

- [ ] **Step 5: Rewrite `src/SDMXerWizard.jl`**

Header docstring: the library gives a model eyes and a checker. `using SDMXer; using HTTP, JSON3, DataFrames, CSV, Statistics, StatsBase, Dates, EzXML, YAML, XLSX; using Logging`. Includes: `SDMXDataSources.jl`, `SDMXDataProfiling.jl`, `SDMXAnonymization.jl`, `SDMXMappingInference.jl`, `SDMXScriptGeneration.jl`, `Tools.jl`, `MCPServer.jl` (the last two are added in Tasks 2 and 7; leave them out until then). Exports: the SDMXer re-exports; data source types and functions; profiling; anonymisation; `MappingCandidate, MappingConfidence, AdvancedMappingResult, InferenceEngine, infer_mappings, infer_advanced_mappings, create_inference_engine, validate_mapping_quality, suggest_value_transformations, analyze_mapping_coverage, learn_from_feedback, fuzzy_match_score, analyze_value_patterns, detect_hierarchical_relationships`; `ScriptTemplate, TransformationStep, GeneratedScript, default_templates, select_template, build_transformation_steps, get_loading_code, validate_generated_script, preview_script_output`. Remove every export that no longer exists (grep each export against `src/`).

- [ ] **Step 6: Trim tests**

`test/runtests.jl`: drop `using HTTP, PromptingTools`; delete the testsets "LLM Integration", "Excel Structure Analysis", "Script Generation" (replace with a "Templates" testset checking `default_templates()` has the four names and `create_standard_template()` sections), "Template Selection" (use `select_template(profile)` and `select_template(simple_csv_profile)`), "Cross-Dataflow", "Script Guidance". Keep the network-backed testsets as they are. `test/simple_test.jl`: delete "LLM Setup" and the `isdefined` checks for removed functions. `docs/make.jl`: drop the four removed pages.

- [ ] **Step 7: Load and run**

Run: `julia --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); using SDMXerWizard'`
Expected: loads without warnings about missing names. Then `julia --project -e 'using Pkg; Pkg.test()'`. Expected: all pass (network tests may skip or fail on an offline machine; note which).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "Remove the internal LLM layer; the model calls the library now"
```

---

### Task 2: Tools module, Session, and load_source

**Files:**
- Create: `src/Tools.jl`
- Modify: `src/SDMXerWizard.jl` (include and export)
- Create: `test/fixtures.jl`, `test/test_tools.jl`, `demo/korea/inbound_visitors.csv`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `Tools.Session()`, `Tools.SchemaEntry`, `Tools.SourceEntry`, `Tools.ToolError(message, hint)`, `Tools.run_tool(session, f, args::AbstractDict) -> Dict{String,Any}`, `Tools.get_schema(session, id)`, `Tools.get_source(session, id)`, `Tools.next_handle!(session, prefix)`, `Tools.jsonable(x)`, `Tools.rows(df, limit)`, `Tools.load_source(session; path, sheet=1, header_row=1, anonymize=false)`.

- [ ] **Step 1: Demo CSV**

`demo/korea/inbound_visitors.csv`:

```
Market,Visitor type,Year,Visitors ('000)
China,Overnight visitors,2015,5984
Japan,Overnight visitors,2015,1837
USA,Overnight visitors,2015,767
China,Same-day visitors,2015,412
Japan,Same-day visitors,2015,96
China,Overnight visitors,2016,6117
Japan,Overnight visitors,2016,2298
USA,Overnight visitors,2016,866
China,Same-day visitors,2016,455
Japan,Same-day visitors,2016,101
```

- [ ] **Step 2: Fixtures**

`test/fixtures.jl` defines `fixture_schema()` (the dry-run schema, with `codelist_id` set to `"CL_AREA"`, `missing`, `"CL_VISITOR_TYPE"`, `"CL_MEASURE"` for the four dimensions and `"CL_UNIT"`, `missing` for the attributes), `fixture_codelists()` returning a DataFrame with columns `codelist_id, code_id, name, parent_code_id`:

| codelist_id | code_id | name |
|---|---|---|
| CL_AREA | CN | China |
| CL_AREA | JP | Japan |
| CL_AREA | US | United States |
| CL_VISITOR_TYPE | OVN | Overnight visitors |
| CL_VISITOR_TYPE | SDV | Same-day visitors |
| CL_MEASURE | ARR | Arrivals |
| CL_UNIT | PS | Persons |

(`parent_code_id` all `missing`), and `const DEMO_CSV = joinpath(@__DIR__, "..", "demo", "korea", "inbound_visitors.csv")`. Move `_dryrun_schema` out of `test_korea_dryrun.jl` into fixtures as `fixture_schema(; with_codelists=false)` and have the dry-run test call it with `with_codelists=false`.

- [ ] **Step 3: Failing test**

`test/test_tools.jl`:

```julia
using Test, JSON3, DataFrames, CSV
using SDMXer, SDMXerWizard
using SDMXerWizard.Tools
include("fixtures.jl")

roundtrip(d) = JSON3.read(JSON3.write(d), Dict{String, Any})

@testset "Session and load_source" begin
    session = Tools.Session()
    out = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV))
    @test out["source_id"] == "source_1"
    @test out["row_count"] == 10
    @test out["column_count"] == 4
    names_out = [c["name"] for c in out["columns"]]
    @test "Visitor type" in names_out
    @test all(c -> length(c["sample_values"]) <= 5, out["columns"])
    @test roundtrip(out)["source_id"] == "source_1"

    bad = Tools.run_tool(session, Tools.load_source, Dict("path" => "nope.csv"))
    @test haskey(bad, "error")
    @test haskey(bad, "hint")

    anon = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV, "anonymize" => true))
    @test anon["source_id"] == "source_2"
    @test !("China" in Tools.get_source(session, "source_2").data[!, "Market"])

    missing_handle = Tools.run_tool(session, (s; source_id) -> Tools.get_source(s, source_id), Dict("source_id" => "source_9"))
    @test occursin("source_1", missing_handle["hint"])
end
```

Add `include("test_tools.jl")` inside a `"Tools"` testset in `runtests.jl`, and `JSON3` and `CSV` to `test/Project.toml` extras if absent.

- [ ] **Step 4: Run, expect failure**

Run: `julia --project -e 'using Pkg; Pkg.test()'`. Expected: `UndefVarError: Tools`.

- [ ] **Step 5: Implement `src/Tools.jl`**

```julia
"""
    SDMXerWizard.Tools

Plain functions a language model calls through MCP. Every function takes a
`Session` first and keyword arguments matching the MCP argument names, and
returns a `Dict{String,Any}` holding only JSON-compatible values.
"""
module Tools

using ..SDMXerWizard: SourceDataProfile, ColumnProfile, MappingCandidate, MappingConfidence,
    AdvancedMappingResult, TransformationStep, ScriptTemplate,
    profile_source_data, read_source_data, anonymize_source_data,
    create_inference_engine, infer_advanced_mappings, validate_mapping_quality,
    fuzzy_match_score, build_transformation_steps, get_loading_code, select_template
using SDMXer: DataflowSchema, extract_dataflow_schema, extract_all_codelists,
    get_required_columns, get_optional_columns, create_validator, validate_sdmx_csv,
    ValidationResult, ValidationIssue
using DataFrames, CSV, JSON3, Dates

# ---------------------------------------------------------------- session

struct SchemaEntry
    schema::DataflowSchema
    codelists::Union{DataFrame, Nothing}
    origin::String
end

mutable struct SourceEntry
    data::DataFrame
    profile::SourceDataProfile
    path::String
    last_mapping::Union{AdvancedMappingResult, Nothing}
end

mutable struct Session
    schemas::Dict{String, SchemaEntry}
    sources::Dict{String, SourceEntry}
    counter::Int
end
Session() = Session(Dict{String, SchemaEntry}(), Dict{String, SourceEntry}(), 0)

function next_handle!(session::Session, prefix::String)
    session.counter += 1
    return prefix * "_" * string(session.counter)
end

struct ToolError <: Exception
    message::String
    hint::String
end
Base.showerror(io::IO, e::ToolError) = print(io, e.message)

function get_schema(session::Session, id)
    key = string(id)
    haskey(session.schemas, key) && return session.schemas[key]
    throw(ToolError("Unknown schema_id " * key,
        "Loaded schemas: " * (isempty(session.schemas) ? "none, call load_schema first" :
                              join(sort(collect(keys(session.schemas))), ", "))))
end

function get_source(session::Session, id)
    key = string(id)
    haskey(session.sources, key) && return session.sources[key]
    throw(ToolError("Unknown source_id " * key,
        "Loaded sources: " * (isempty(session.sources) ? "none, call load_source first" :
                              join(sort(collect(keys(session.sources))), ", "))))
end

# ---------------------------------------------------------------- dispatch

"""
    run_tool(session, f, args) -> Dict{String,Any}

Call `f(session; args...)` and turn any exception into an error dict.
"""
function run_tool(session::Session, f, args::AbstractDict)
    kwargs = Dict{Symbol, Any}(Symbol(string(k)) => plain(v) for (k, v) in pairs(args))
    try
        return f(session; kwargs...)
    catch e
        e isa ToolError && return Dict{String, Any}("error" => e.message, "hint" => e.hint)
        return Dict{String, Any}("error" => sprint(showerror, e),
                                 "hint" => "Check the arguments; call the tool again with corrected values.")
    end
end

# JSON3 objects and arrays become plain Dicts and Vectors before reaching a tool.
plain(x) = x
plain(x::AbstractDict) = Dict{String, Any}(string(k) => plain(v) for (k, v) in pairs(x))
plain(x::AbstractVector) = Any[plain(v) for v in x]
plain(x::AbstractString) = String(x)

# ---------------------------------------------------------------- serialisation

const MAX_SAMPLE_VALUES = 5
const MAX_SAMPLE_LENGTH = 60

truncate_str(s, n=MAX_SAMPLE_LENGTH) = length(s) <= n ? String(s) : first(s, n - 1) * "…"

jsonable(x::Nothing) = nothing
jsonable(x::Missing) = nothing
jsonable(x::Bool) = x
jsonable(x::Integer) = Int(x)
jsonable(x::AbstractFloat) = isfinite(x) ? round(Float64(x); sigdigits=4) : nothing
jsonable(x::AbstractString) = String(x)
jsonable(x::Symbol) = string(x)
jsonable(x::Type) = string(x)
jsonable(x::Enum) = string(x)
jsonable(x::Union{Date, DateTime}) = string(x)
jsonable(x::AbstractVector) = Any[jsonable(v) for v in x]
jsonable(x::Tuple) = Any[jsonable(v) for v in x]
jsonable(x::AbstractDict) = Dict{String, Any}(string(k) => jsonable(v) for (k, v) in pairs(x))
jsonable(x::NamedTuple) = Dict{String, Any}(string(k) => jsonable(v) for (k, v) in pairs(x))
jsonable(x) = string(x)

function rows(df::DataFrame, limit::Integer)
    n = min(nrow(df), limit)
    return Any[Dict{String, Any}(string(c) => jsonable(df[i, c]) for c in names(df)) for i in 1:n]
end

function jsonable(c::ColumnProfile)
    samples = [truncate_str(string(v)) for v in Iterators.take(c.sample_values, MAX_SAMPLE_VALUES)]
    return Dict{String, Any}(
        "name" => c.name,
        "type" => string(c.type),
        "missing_count" => c.missing_count,
        "unique_count" => c.unique_count,
        "sample_values" => samples,
        "is_categorical" => c.is_categorical,
        "is_temporal" => c.is_temporal,
        "temporal_format" => jsonable(c.temporal_format),
        "numeric_stats" => jsonable(c.numeric_stats))
end

function jsonable(p::SourceDataProfile)
    return Dict{String, Any}(
        "path" => p.file_path,
        "file_type" => p.file_type,
        "row_count" => p.row_count,
        "column_count" => p.column_count,
        "columns" => Any[jsonable(c) for c in p.columns],
        "data_quality_score" => jsonable(p.data_quality_score),
        "suggested_key_columns" => p.suggested_key_columns,
        "suggested_value_columns" => p.suggested_value_columns,
        "suggested_time_columns" => p.suggested_time_columns)
end

# ---------------------------------------------------------------- load_source

"""
    load_source(session; path, sheet=1, header_row=1, anonymize=false)

Read a CSV or Excel file, profile it, store it in the session and return the
handle plus a compact profile.
"""
function load_source(session::Session; path, sheet=1, header_row=1, anonymize=false)
    isfile(path) || throw(ToolError("File not found: " * string(path),
        "Give an absolute path to an existing .csv, .xlsx or .xls file."))
    df = read_source_data(String(path); sheet=sheet, header_row=Int(header_row))
    profile = profile_source_data(df, String(path))
    if anonymize
        df = anonymize_source_data(df, profile)
        profile = profile_source_data(df, String(path))
    end
    id = next_handle!(session, "source")
    session.sources[id] = SourceEntry(df, profile, String(path), nothing)
    out = jsonable(profile)
    out["source_id"] = id
    out["anonymized"] = Bool(anonymize)
    return out
end

end # module Tools
```

Add `include("Tools.jl")` after `SDMXScriptGeneration.jl` in `SDMXerWizard.jl` and `export Tools`.

- [ ] **Step 6: Run, expect pass**

Run: `julia --project -e 'using Pkg; Pkg.test()'`. Expected: the "Session and load_source" testset passes.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Add the Tools session and load_source"
```

---

### Task 3: load_schema, register_schema!, lookup_codes

**Files:**
- Modify: `src/Tools.jl`, `test/test_tools.jl`

**Interfaces:**
- Produces: `Tools.register_schema!(session, schema::DataflowSchema, codelists::Union{DataFrame,Nothing}; origin="") -> String`, `Tools.schema_summary(entry, id) -> Dict`, `Tools.load_schema(session; url, with_codelists=true)`, `Tools.lookup_codes(session; schema_id, dimension, query=nothing, limit=20)`, `Tools.codelist_for(entry, column_id) -> Union{String,Nothing}`, `Tools.codes_for(entry, codelist_id) -> DataFrame`.

- [ ] **Step 1: Failing test**

```julia
@testset "Schema registration and code lookup" begin
    session = Tools.Session()
    id = Tools.register_schema!(session, fixture_schema(), fixture_codelists(); origin="fixture")
    @test id == "schema_1"
    summary = Tools.schema_summary(session.schemas[id], id)
    @test summary["schema_id"] == id
    @test [d["id"] for d in summary["dimensions"]] == ["REF_AREA", "COUNTERPART_AREA", "VISITOR_TYPE", "MEASURE"]
    @test summary["time_dimension"]["id"] == "TIME_PERIOD"
    @test "OBS_VALUE" in summary["required_columns"]
    @test summary["codelists"]["CL_AREA"] == 3
    @test roundtrip(summary)["schema_id"] == id

    all_codes = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "VISITOR_TYPE"))
    @test all_codes["total_codes"] == 2
    @test [m["code"] for m in all_codes["matches"]] == ["OVN", "SDV"]

    hit = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "REF_AREA", "query" => "japan"))
    @test hit["matches"][1]["code"] == "JP"

    attr = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "UNIT_MEASURE"))
    @test attr["codelist_id"] == "CL_UNIT"

    none = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "COUNTERPART_AREA"))
    @test haskey(none, "error")

    limited = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "REF_AREA", "limit" => 1))
    @test length(limited["matches"]) == 1
    @test limited["truncated"] == true
end
```

- [ ] **Step 2: Run, expect failure** (`UndefVarError: register_schema!`).

- [ ] **Step 3: Implement**

Append to `src/Tools.jl` before `end # module Tools`:

```julia
# ---------------------------------------------------------------- schemas

function register_schema!(session::Session, schema::DataflowSchema,
                          codelists::Union{DataFrame, Nothing}; origin::String="")
    id = next_handle!(session, "schema")
    session.schemas[id] = SchemaEntry(schema, codelists, origin)
    return id
end

function codelist_for(entry::SchemaEntry, column_id::AbstractString)
    schema = entry.schema
    for row in eachrow(schema.dimensions)
        row.dimension_id == column_id && return ismissing(row.codelist_id) ? nothing : String(row.codelist_id)
    end
    for row in eachrow(schema.attributes)
        row.attribute_id == column_id && return ismissing(row.codelist_id) ? nothing : String(row.codelist_id)
    end
    return nothing
end

function codes_for(entry::SchemaEntry, codelist_id::AbstractString)
    entry.codelists === nothing && return DataFrame()
    return entry.codelists[entry.codelists.codelist_id .== codelist_id, :]
end

function schema_summary(entry::SchemaEntry, id::String)
    schema = entry.schema
    info = schema.dataflow_info
    getfield_or(nt, k) = hasproperty(nt, k) ? jsonable(getproperty(nt, k)) : nothing
    dims = Any[Dict{String, Any}(
        "id" => r.dimension_id, "position" => r.position,
        "codelist_id" => jsonable(r.codelist_id), "is_time_dimension" => r.is_time_dimension)
        for r in eachrow(schema.dimensions)]
    td = schema.time_dimension
    time_dim = td === nothing ? nothing : Dict{String, Any}(
        "id" => td.dimension_id, "position" => td.position, "data_type" => jsonable(td.data_type))
    attrs = Any[Dict{String, Any}(
        "id" => r.attribute_id, "assignment_status" => jsonable(r.assignment_status),
        "relationship" => jsonable(r.relationship), "codelist_id" => jsonable(r.codelist_id))
        for r in eachrow(schema.attributes)]
    measures = Any[Dict{String, Any}("id" => r.measure_id, "data_type" => jsonable(r.data_type))
        for r in eachrow(schema.measures)]
    counts = Dict{String, Any}()
    if entry.codelists !== nothing && nrow(entry.codelists) > 0
        for g in groupby(entry.codelists, :codelist_id)
            counts[String(first(g.codelist_id))] = nrow(g)
        end
    end
    return Dict{String, Any}(
        "schema_id" => id,
        "origin" => entry.origin,
        "dataflow" => Dict{String, Any}("id" => getfield_or(info, :id), "agency" => getfield_or(info, :agency),
            "version" => getfield_or(info, :version), "name" => getfield_or(info, :name)),
        "dimensions" => dims,
        "time_dimension" => time_dim,
        "attributes" => attrs,
        "measures" => measures,
        "required_columns" => get_required_columns(schema),
        "optional_columns" => get_optional_columns(schema),
        "codelists" => counts,
        "codelists_loaded" => entry.codelists !== nothing)
end

"""
    load_schema(session; url, with_codelists=true)

Fetch a dataflow structure (URL or local SDMx-ML file), store it and return a
summary. Codes are fetched too unless `with_codelists` is false.
"""
function load_schema(session::Session; url, with_codelists=true)
    schema = extract_dataflow_schema(String(url))
    codelists = nothing
    if with_codelists
        try
            codelists = extract_all_codelists(String(url))
        catch e
            @warn "Codelists could not be fetched" url exception=e
        end
    end
    id = register_schema!(session, schema, codelists; origin=String(url))
    return schema_summary(session.schemas[id], id)
end

"""
    lookup_codes(session; schema_id, dimension, query=nothing, limit=20)

List or search the codes of the codelist behind a dimension or attribute.
"""
function lookup_codes(session::Session; schema_id, dimension, query=nothing, limit=20)
    entry = get_schema(session, schema_id)
    cl = codelist_for(entry, String(dimension))
    cl === nothing && throw(ToolError("No codelist behind " * string(dimension),
        "Only dimensions and attributes with a codelist_id in load_schema can be looked up."))
    entry.codelists === nothing && throw(ToolError("Schema " * string(schema_id) * " was loaded without codelists",
        "Call load_schema again with with_codelists=true."))
    codes = codes_for(entry, cl)
    total = nrow(codes)
    scored = Tuple{Float64, DataFrameRow}[]
    if query === nothing || isempty(string(query))
        for r in eachrow(codes)
            push!(scored, (1.0, r))
        end
    else
        q = lowercase(string(query))
        for r in eachrow(codes)
            name = ismissing(r.name) ? "" : String(r.name)
            score = occursin(q, lowercase(r.code_id)) || occursin(q, lowercase(name)) ? 1.0 :
                    max(fuzzy_match_score(q, r.code_id), fuzzy_match_score(q, name))
            score >= 0.3 && push!(scored, (score, r))
        end
        sort!(scored; by=first, rev=true)
    end
    lim = Int(limit)
    kept = scored[1:min(lim, length(scored))]
    matches = Any[Dict{String, Any}("code" => r.code_id, "name" => jsonable(r.name),
        "parent" => hasproperty(r, :parent_code_id) ? jsonable(r.parent_code_id) : nothing,
        "score" => jsonable(s)) for (s, r) in kept]
    return Dict{String, Any}("dimension" => String(dimension), "codelist_id" => cl,
        "total_codes" => total, "matches" => matches, "truncated" => length(scored) > lim)
end
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add load_schema and lookup_codes tools"`

---

### Task 4: infer_mappings tool

**Files:**
- Modify: `src/Tools.jl`, `test/test_tools.jl`

**Interfaces:**
- Produces: `Tools.infer_mappings(session; source_id, schema_id, method="advanced", confidence_threshold=0.3)`, `Tools.preload_codelists!(engine, entry)`, `Tools.jsonable(::MappingCandidate)`, `Tools.jsonable(::AdvancedMappingResult)`, `Tools.mapping_report(result, entry) -> Dict`.

- [ ] **Step 1: Failing test**

```julia
@testset "infer_mappings tool" begin
    session = Tools.Session()
    sid = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV))["source_id"]
    schid = Tools.register_schema!(session, fixture_schema(), fixture_codelists(); origin="fixture")
    for method in ["heuristic", "fuzzy", "advanced"]
        out = Tools.run_tool(session, Tools.infer_mappings,
            Dict("source_id" => sid, "schema_id" => schid, "method" => method, "confidence_threshold" => 0.2))
        @test !haskey(out, "error")
        @test roundtrip(out)["method"] == method
        targets = [m["target_column"] for m in out["mappings"]]
        @test "VISITOR_TYPE" in targets
        vt = only(filter(m -> m["target_column"] == "VISITOR_TYPE", out["mappings"]))
        @test vt["candidates"][1]["source_column"] == "Visitor type"
        @test haskey(out, "unmapped_required_columns")
        @test out["warnings"] isa Vector
    end
    @test Tools.get_source(session, sid).last_mapping isa AdvancedMappingResult
    bad = Tools.run_tool(session, Tools.infer_mappings, Dict("source_id" => sid, "schema_id" => schid, "method" => "llm"))
    @test occursin("heuristic", bad["hint"])
end
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement**

```julia
# ---------------------------------------------------------------- mappings

const MAPPING_METHODS = ("heuristic", "fuzzy", "advanced")

function jsonable(m::MappingCandidate)
    return Dict{String, Any}(
        "source_column" => m.source_column,
        "target_column" => m.target_column,
        "confidence" => jsonable(m.confidence_score),
        "confidence_level" => string(m.confidence_level),
        "match_type" => m.match_type,
        "evidence" => jsonable(m.evidence),
        "suggested_transformation" => jsonable(m.suggested_transformation),
        "notes" => m.validation_notes)
end

function preload_codelists!(engine, entry::SchemaEntry)
    entry.codelists === nothing && return engine
    for g in groupby(entry.codelists, :codelist_id)
        engine.codelists_data[String(first(g.codelist_id))] = DataFrame(g)
    end
    return engine
end

function mapping_report(result::AdvancedMappingResult, entry::SchemaEntry)
    schema = entry.schema
    order = String[]
    append!(order, schema.dimensions.dimension_id)
    schema.time_dimension === nothing || push!(order, schema.time_dimension.dimension_id)
    append!(order, schema.measures.measure_id)
    append!(order, schema.attributes.attribute_id)
    grouped = Dict{String, Vector{MappingCandidate}}()
    for m in result.mappings
        push!(get!(grouped, m.target_column, MappingCandidate[]), m)
    end
    mappings = Any[]
    for t in order
        haskey(grouped, t) || continue
        cands = sort(grouped[t]; by=c -> c.confidence_score, rev=true)
        push!(mappings, Dict{String, Any}("target_column" => t,
            "codelist_id" => jsonable(codelist_for(entry, t)),
            "candidates" => Any[jsonable(c) for c in cands]))
    end
    quality = validate_mapping_quality(result, schema)
    return Dict{String, Any}(
        "mappings" => mappings,
        "unmapped_source_columns" => result.unmapped_source_columns,
        "unmapped_target_columns" => result.unmapped_target_columns,
        "unmapped_required_columns" => intersect(result.unmapped_target_columns, get_required_columns(schema)),
        "quality_score" => jsonable(result.quality_score),
        "recommendations" => result.recommendations,
        "warnings" => vcat(quality["critical_issues"], quality["warnings"]))
end

"""
    infer_mappings(session; source_id, schema_id, method="advanced", confidence_threshold=0.3)

Rank source columns against the target columns of a schema. `heuristic` uses
column names only, `fuzzy` adds statistical analysis, `advanced` adds codelist
value matching.
"""
function infer_mappings(session::Session; source_id, schema_id, method="advanced", confidence_threshold=0.3)
    m = String(method)
    m in MAPPING_METHODS || throw(ToolError("Unknown method " * m,
        "method must be one of: " * join(MAPPING_METHODS, ", ")))
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    engine = create_inference_engine(min_confidence=Float64(confidence_threshold),
        use_statistical_analysis=(m != "heuristic"), use_value_matching=(m == "advanced"))
    m == "advanced" && preload_codelists!(engine, entry)
    result = infer_advanced_mappings(engine, src.profile, entry.schema, src.data)
    src.last_mapping = result
    out = mapping_report(result, entry)
    out["method"] = m
    out["source_id"] = String(source_id)
    out["schema_id"] = String(schema_id)
    return out
end
```

- [ ] **Step 4: Run, expect pass.** If "Visitor type" is not the top candidate for `VISITOR_TYPE` under `heuristic`, inspect `fuzzy_match_score("Visitor type", "VISITOR_TYPE")` and adjust the assertion to `in` the candidate list rather than position 1, noting the reason in the test.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add the infer_mappings tool"`

---

### Task 5: transformation_plan tool

**Files:**
- Modify: `src/Tools.jl`, `test/test_tools.jl`

**Interfaces:**
- Produces: `Tools.transformation_plan(session; source_id, schema_id, mappings=nothing)`, `Tools.choose_mappings(result, overrides) -> AdvancedMappingResult`, `Tools.recodings(entry, chosen, data) -> Vector`, `Tools.SCRIPT_CONTRACT::String`, `Tools.jsonable(::TransformationStep)`.

- [ ] **Step 1: Failing test**

```julia
@testset "transformation_plan tool" begin
    session = Tools.Session()
    sid = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV))["source_id"]
    schid = Tools.register_schema!(session, fixture_schema(), fixture_codelists(); origin="fixture")
    plan = Tools.run_tool(session, Tools.transformation_plan, Dict("source_id" => sid, "schema_id" => schid))
    @test !haskey(plan, "error")
    @test plan["steps"][1]["operation_type"] == "read"
    @test occursin("CSV.read", plan["loading_code"])
    @test plan["template"]["name"] in ("simple_csv", "standard_transformation")
    @test occursin("result", plan["contract"])
    @test roundtrip(plan)["contract"] == plan["contract"]

    overridden = Tools.run_tool(session, Tools.transformation_plan,
        Dict("source_id" => sid, "schema_id" => schid,
             "mappings" => Dict("REF_AREA" => "Market", "VISITOR_TYPE" => "Visitor type",
                                "TIME_PERIOD" => "Year", "OBS_VALUE" => "Visitors ('000)")))
    @test Set(m["target_column"] for m in overridden["chosen_mappings"]) ⊇ Set(["REF_AREA", "VISITOR_TYPE", "TIME_PERIOD", "OBS_VALUE"])
    rec = only(filter(r -> r["target_column"] == "VISITOR_TYPE", overridden["recodings"]))
    ov = only(filter(v -> v["source_value"] == "Overnight visitors", rec["values"]))
    @test ov["candidates"][1]["code"] == "OVN"
    area = only(filter(r -> r["target_column"] == "REF_AREA", overridden["recodings"]))
    @test any(v -> v["source_value"] == "USA", area["values"])
    @test isempty(filter(r -> r["target_column"] == "TIME_PERIOD", overridden["recodings"]))

    bad = Tools.run_tool(session, Tools.transformation_plan,
        Dict("source_id" => sid, "schema_id" => schid, "mappings" => Dict("REF_AREA" => "Nope")))
    @test occursin("Nope", bad["error"])
end
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement**

```julia
# ---------------------------------------------------------------- plan

const SCRIPT_CONTRACT = """
The script is evaluated in a fresh module where `source` is the loaded source DataFrame
and the packages DataFrames, CSV and Dates are already in scope, together with SDMXer.
Do not read the file again. The script must assign a DataFrame to a variable named
`result` whose columns are the SDMx columns of the target schema (dimensions in order,
TIME_PERIOD, OBS_VALUE, then attributes). Code values must be codelist ids, not labels.
Use string concatenation rather than string interpolation."""

function jsonable(s::TransformationStep)
    return Dict{String, Any}(
        "name" => s.step_name, "operation_type" => s.operation_type,
        "function" => s.tidier_function, "description" => s.description,
        "source_columns" => s.source_columns, "target_columns" => s.target_columns,
        "logic" => s.transformation_logic, "validation_checks" => s.validation_checks,
        "comments" => s.comments)
end

function choose_mappings(result::AdvancedMappingResult, overrides::Dict{String, String}, entry::SchemaEntry, src::SourceEntry)
    best = Dict{String, MappingCandidate}()
    for m in result.mappings
        if !haskey(best, m.target_column) || m.confidence_score > best[m.target_column].confidence_score
            best[m.target_column] = m
        end
    end
    for (target, source) in overrides
        source in names(src.data) || throw(ToolError("Source column " * source * " does not exist",
            "Source columns: " * join(names(src.data), ", ")))
        best[target] = MappingCandidate(source, target, 1.0, MappingConfidence(0), "user",
            Dict{String, Any}("origin" => "override"), nothing, String[])
    end
    chosen = collect(values(best))
    mapped_sources = Set(m.source_column for m in chosen)
    mapped_targets = Set(m.target_column for m in chosen)
    all_targets = vcat(get_required_columns(entry.schema), get_optional_columns(entry.schema))
    return AdvancedMappingResult(chosen, result.coverage_analysis,
        [c for c in names(src.data) if !(c in mapped_sources)],
        [t for t in all_targets if !(t in mapped_targets)],
        result.quality_score, result.recommendations, result.transformation_complexity)
end

function recodings(entry::SchemaEntry, chosen::AdvancedMappingResult, data::DataFrame)
    out = Any[]
    entry.codelists === nothing && return out
    for m in chosen.mappings
        cl = codelist_for(entry, m.target_column)
        cl === nothing && continue
        codes = codes_for(entry, cl)
        nrow(codes) == 0 && continue
        ids = Set(codes.code_id)
        values = unique(string.(skipmissing(data[!, m.source_column])))
        pending = [v for v in values if !(v in ids)]
        isempty(pending) && continue
        entries = Any[]
        for v in Iterators.take(pending, 50)
            lv = lowercase(v)
            scored = [(max(occursin(lv, lowercase(r.code_id)) || (!ismissing(r.name) && occursin(lv, lowercase(r.name))) ? 1.0 : 0.0,
                           fuzzy_match_score(v, r.code_id),
                           ismissing(r.name) ? 0.0 : fuzzy_match_score(v, r.name)), r) for r in eachrow(codes)]
            sort!(scored; by=first, rev=true)
            push!(entries, Dict{String, Any}("source_value" => v,
                "candidates" => Any[Dict{String, Any}("code" => r.code_id, "name" => jsonable(r.name), "score" => jsonable(s))
                                    for (s, r) in Iterators.take(scored, 3) if s > 0.0]))
        end
        push!(out, Dict{String, Any}("target_column" => m.target_column, "source_column" => m.source_column,
            "codelist_id" => cl, "values" => entries, "truncated" => length(pending) > 50))
    end
    return out
end

"""
    transformation_plan(session; source_id, schema_id, mappings=nothing)

Turn the chosen mappings into ordered transformation steps, a loading snippet,
a template skeleton, the recodings still to decide, and the script contract.
`mappings` is an optional object `{target_column: source_column}` that overrides
the inferred candidates.
"""
function transformation_plan(session::Session; source_id, schema_id, mappings=nothing)
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    if src.last_mapping === nothing
        infer_mappings(session; source_id=source_id, schema_id=schema_id)
    end
    overrides = Dict{String, String}()
    if mappings !== nothing
        mappings isa AbstractDict || throw(ToolError("mappings must be an object", "Use {\"TARGET\": \"source column\"}"))
        for (k, v) in pairs(mappings)
            overrides[string(k)] = string(v)
        end
    end
    chosen = choose_mappings(src.last_mapping, overrides, entry, src)
    steps = build_transformation_steps(chosen, src.profile, entry.schema)
    template = select_template(src.profile)
    return Dict{String, Any}(
        "source_id" => String(source_id), "schema_id" => String(schema_id),
        "chosen_mappings" => Any[jsonable(m) for m in sort(chosen.mappings; by=m -> m.target_column)],
        "unmapped_target_columns" => chosen.unmapped_target_columns,
        "unmapped_required_columns" => intersect(chosen.unmapped_target_columns, get_required_columns(entry.schema)),
        "steps" => Any[jsonable(s) for s in steps],
        "loading_code" => get_loading_code(src.profile),
        "template" => Dict{String, Any}("name" => template.template_name, "description" => template.description,
            "sections" => jsonable(template.template_sections), "example" => template.example_code),
        "recodings" => recodings(entry, chosen, src.data),
        "contract" => SCRIPT_CONTRACT)
end
```

`MappingConfidence(0)` must be the highest level; check the enum in `SDMXMappingInference.jl` and use the named constant (`HIGH` or `VERY_HIGH`) instead.

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add the transformation_plan tool"`

---

### Task 6: run_script and validate_csv

**Files:**
- Modify: `src/Tools.jl`, `test/test_tools.jl`

**Interfaces:**
- Produces: `Tools.run_script(session; source_id, schema_id, script, output_path=nothing, preview_rows=10)`, `Tools.validate_csv(session; path, schema_id, preview_rows=10)`, `Tools.sandbox_run(script::String, source::DataFrame) -> DataFrame`, `Tools.validation_report(schema, df, name) -> Dict`, `Tools.jsonable(::ValidationResult)`, `Tools.jsonable(::ValidationIssue)`.

- [ ] **Step 1: Failing test**

```julia
const GOOD_SCRIPT = """
area = Dict("China" => "CN", "Japan" => "JP", "USA" => "US")
vtype = Dict("Overnight visitors" => "OVN", "Same-day visitors" => "SDV")
result = DataFrame(
    REF_AREA = [area[string(x)] for x in source[!, "Market"]],
    COUNTERPART_AREA = fill("KR", nrow(source)),
    VISITOR_TYPE = [vtype[string(x)] for x in source[!, "Visitor type"]],
    MEASURE = fill("ARR", nrow(source)),
    TIME_PERIOD = string.(source[!, "Year"]),
    OBS_VALUE = Float64.(source[!, "Visitors ('000)"]) .* 1000,
    UNIT_MEASURE = fill("PS", nrow(source)))
"""

@testset "run_script and validate_csv" begin
    session = Tools.Session()
    sid = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV))["source_id"]
    schid = Tools.register_schema!(session, fixture_schema(), fixture_codelists(); origin="fixture")
    outpath = joinpath(mktempdir(), "out.csv")
    ok = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => GOOD_SCRIPT, "output_path" => outpath, "preview_rows" => 3))
    @test !haskey(ok, "error")
    @test ok["result_rows"] == 10
    @test length(ok["preview"]) == 3
    @test ok["validation"]["compliance_status"] in ("compliant", "minor_issues", "major_issues", "non_compliant")
    @test ok["validation"]["issues"] isa Vector
    @test isfile(outpath)
    @test roundtrip(ok)["result_rows"] == 10

    throws = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "x = 1\nerror(\"boom\")\nresult = source"))
    @test occursin("boom", throws["error"])
    @test occursin("line 2", throws["error"])

    noresult = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "x = source"))
    @test occursin("result", noresult["error"])

    parsefail = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "result = DataFrame(("))
    @test haskey(parsefail, "error")

    # the sandbox must not leak into the stored source
    @test "Market" in names(Tools.get_source(session, sid).data)

    vc = Tools.run_tool(session, Tools.validate_csv, Dict("path" => outpath, "schema_id" => schid))
    @test vc["validation"]["compliance_status"] == ok["validation"]["compliance_status"]
end
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement**

```julia
# ---------------------------------------------------------------- validation

function jsonable(i::ValidationIssue)
    return Dict{String, Any}(
        "rule_id" => i.rule_id, "severity" => string(i.severity), "message" => i.message,
        "location" => i.location, "affected_row_count" => length(i.affected_rows),
        "affected_rows" => collect(Iterators.take(i.affected_rows, 10)),
        "suggested_fix" => i.suggested_fix, "auto_fixable" => i.auto_fixable)
end

function jsonable(v::ValidationResult)
    return Dict{String, Any}(
        "dataset_name" => v.dataset_name, "compliance_status" => v.compliance_status,
        "overall_score" => jsonable(v.overall_score), "total_rows" => v.total_rows,
        "total_columns" => v.total_columns, "issues" => Any[jsonable(i) for i in v.issues],
        "recommendations" => v.recommendations, "statistics" => jsonable(v.statistics))
end

function validation_report(schema::DataflowSchema, df::DataFrame, name::String, preview_rows::Integer)
    validator = create_validator(schema)
    vr = validate_sdmx_csv(validator, df, name)
    return Dict{String, Any}("validation" => jsonable(vr), "preview" => rows(df, preview_rows),
        "result_rows" => nrow(df), "result_columns" => names(df))
end

# ---------------------------------------------------------------- sandbox

const SANDBOX_MODULES = (DataFrames, CSV, Dates)

function sandbox_run(script::String, source::DataFrame)
    m = Module(:SDMXerScript)
    for pkg in SANDBOX_MODULES
        Core.eval(m, Expr(:const, Expr(:(=), nameof(pkg), pkg)))
        Core.eval(m, Expr(:using, Expr(:., :., nameof(pkg))))
    end
    Core.eval(m, Expr(:const, Expr(:(=), :SDMXer, SDMXer)))
    Core.eval(m, Expr(:using, Expr(:., :., :SDMXer)))
    Core.eval(m, Expr(:(=), :source, copy(source)))
    parsed = try
        Meta.parseall(script; filename="script.jl")
    catch e
        throw(ToolError("Script failed to parse: " * sprint(showerror, e), SCRIPT_CONTRACT))
    end
    line = 0
    for ex in parsed.args
        if ex isa LineNumberNode
            line = ex.line
            continue
        end
        if ex isa Expr && ex.head == :error
            throw(ToolError("Script failed to parse at line " * string(line) * ": " * string(ex.args[1]), SCRIPT_CONTRACT))
        end
        try
            Core.eval(m, ex)
        catch e
            throw(ToolError("Script failed at line " * string(line) * ": " * sprint(showerror, e), SCRIPT_CONTRACT))
        end
    end
    isdefined(m, :result) || throw(ToolError("The script did not define `result`", SCRIPT_CONTRACT))
    result = getfield(m, :result)
    result isa DataFrame || throw(ToolError("`result` is a " * string(typeof(result)) * ", expected a DataFrame", SCRIPT_CONTRACT))
    return result
end

"""
    run_script(session; source_id, schema_id, script, output_path=nothing, preview_rows=10)

Evaluate a transformation script against a loaded source, validate the
resulting DataFrame against the schema, and optionally write it as CSV.
"""
function run_script(session::Session; source_id, schema_id, script, output_path=nothing, preview_rows=10)
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    result = sandbox_run(String(script), src.data)
    out = validation_report(entry.schema, result, basename(src.path), Int(preview_rows))
    if output_path !== nothing && !isempty(string(output_path))
        CSV.write(String(output_path), result)
        out["output_path"] = String(output_path)
    else
        out["output_path"] = nothing
    end
    out["source_id"] = String(source_id)
    out["schema_id"] = String(schema_id)
    return out
end

"""
    validate_csv(session; path, schema_id, preview_rows=10)

Validate an SDMx-CSV file produced elsewhere against a loaded schema.
"""
function validate_csv(session::Session; path, schema_id, preview_rows=10)
    isfile(path) || throw(ToolError("File not found: " * string(path), "Give the path of an existing CSV file."))
    entry = get_schema(session, schema_id)
    df = CSV.read(String(path), DataFrame)
    out = validation_report(entry.schema, df, basename(String(path)), Int(preview_rows))
    out["path"] = String(path)
    out["schema_id"] = String(schema_id)
    return out
end
```

`Module(:SDMXerScript)` imports `Base` by default, so `Dict`, `string`, `nrow` (after `using .DataFrames`) all resolve. Add `using SDMXer` (the module, for the `SDMXer` binding) at the top of `Tools.jl`: `import SDMXer`.

- [ ] **Step 4: Run, expect pass.** If `validate_sdmx_csv` rejects a column type (for example `TIME_PERIOD` as `String` vs `Int`), that is a validation issue in the returned report, which is the expected behaviour; only the status string is asserted.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add run_script and validate_csv tools"`

---

### Task 7: MCP server

**Files:**
- Create: `src/MCPServer.jl`, `.mcp.json`
- Modify: `src/SDMXerWizard.jl`, `test/test_tools.jl`

**Interfaces:**
- Produces: `SDMXerWizard.mcp_tools(session::Tools.Session) -> Vector{MCPTool}`, `SDMXerWizard.serve_mcp(; name="sdmxer-wizard", version=...)`, `SDMXerWizard.TOOL_SPECS`, `SDMXerWizard.SERVER_INSTRUCTIONS`.

- [ ] **Step 1: Failing test**

```julia
@testset "MCP wiring" begin
    session = Tools.Session()
    tools = SDMXerWizard.mcp_tools(session)
    @test Set(t.name for t in tools) == Set(["load_schema", "lookup_codes", "load_source", "infer_mappings",
                                             "transformation_plan", "run_script", "validate_csv"])
    for t in tools
        @test t.input_schema["type"] == "object"
        @test haskey(t.input_schema, "required")
    end
    load_tool = only(filter(t -> t.name == "load_source", tools))
    content = load_tool.handler(Dict{String, Any}("path" => DEMO_CSV))
    @test content isa ModelContextProtocol.TextContent
    parsed = JSON3.read(content.text, Dict{String, Any})
    @test parsed["source_id"] == "source_1"

    # a tool body that prints must not corrupt the returned content
    noisy = SDMXerWizard.tool_handler(session, (s; x) -> (println("stray output"); Dict{String, Any}("x" => x)))
    c = noisy(Dict{String, Any}("x" => 1))
    @test JSON3.read(c.text, Dict{String, Any})["x"] == 1
end

@testset "MCP stdio round trip" begin
    project = normpath(joinpath(@__DIR__, ".."))
    cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no -e "using SDMXerWizard; serve_mcp()"`
    proc = open(cmd, "r+")
    init = Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize",
        "params" => Dict("protocolVersion" => "2025-06-18", "capabilities" => Dict(),
                         "clientInfo" => Dict("name" => "test", "version" => "0")))
    println(proc, JSON3.write(init)); flush(proc)
    reply = JSON3.read(readline(proc), Dict{String, Any})
    @test reply["id"] == 1
    @test haskey(reply["result"], "serverInfo")
    println(proc, JSON3.write(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"))); flush(proc)
    println(proc, JSON3.write(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list"))); flush(proc)
    listing = JSON3.read(readline(proc), Dict{String, Any})
    @test length(listing["result"]["tools"]) == 7
    close(proc)
end
```

Add `using ModelContextProtocol` to the test file header and the package to `test/Project.toml` if the test env needs it explicitly.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `src/MCPServer.jl`**

```julia
using ModelContextProtocol
using Logging

const SERVER_INSTRUCTIONS = """
SDMXerWizard maps a source data file (CSV or Excel) to an SDMx dataflow and validates the result.
Typical order: load_schema, load_source, infer_mappings, lookup_codes as needed, transformation_plan,
then write a Julia script that follows the returned contract and submit it with run_script.
Iterate on the script until the validation report is compliant. Handles (schema_1, source_1) refer to
objects held in this server's session; they do not survive a restart."""

prop(type, description; extra...) = merge(Dict{String, Any}("type" => type, "description" => description),
                                          Dict{String, Any}(string(k) => v for (k, v) in pairs(extra)))

object_schema(props::Vector{Pair{String, Dict{String, Any}}}, required::Vector{String}) =
    Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}(props...), "required" => required)

const TOOL_SPECS = [
    (name = "load_schema", fn = Tools.load_schema,
     description = "Fetch an SDMx dataflow structure from a .Stat URL or a local SDMx-ML file and store it in the session. Returns schema_id, dimensions, attributes, measures, required columns and codelist sizes.",
     schema = object_schema(["url" => prop("string", "Dataflow URL (for example a .Stat Data Explorer developer API link with references=all) or path to a structure XML file"),
                             "with_codelists" => prop("boolean", "Also fetch the codelists so lookup_codes and value matching work"; default = true)],
                            ["url"])),
    (name = "lookup_codes", fn = Tools.lookup_codes,
     description = "List or search the codes behind a dimension or attribute of a loaded schema.",
     schema = object_schema(["schema_id" => prop("string", "Handle returned by load_schema"),
                             "dimension" => prop("string", "Dimension or attribute id, for example REF_AREA"),
                             "query" => prop("string", "Optional case-insensitive text matched against code ids and names"),
                             "limit" => prop("integer", "Maximum number of codes to return"; default = 20)],
                            ["schema_id", "dimension"])),
    (name = "load_source", fn = Tools.load_source,
     description = "Read a CSV or Excel file, profile its columns and store it in the session. Returns source_id and a compact column profile.",
     schema = object_schema(["path" => prop("string", "Absolute path to a .csv, .xlsx or .xls file"),
                             "sheet" => prop("integer", "Excel sheet number"; default = 1),
                             "header_row" => prop("integer", "Row holding the column names"; default = 1),
                             "anonymize" => prop("boolean", "Replace values with synthetic ones before anything is returned"; default = false)],
                            ["path"])),
    (name = "infer_mappings", fn = Tools.infer_mappings,
     description = "Rank source columns against the target columns of a schema and report unmapped required columns.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "method" => prop("string", "heuristic (names only), fuzzy (names and statistics) or advanced (adds codelist value matching)"; enum = ["heuristic", "fuzzy", "advanced"], default = "advanced"),
                             "confidence_threshold" => prop("number", "Drop candidates below this confidence"; default = 0.3)],
                            ["source_id", "schema_id"])),
    (name = "transformation_plan", fn = Tools.transformation_plan,
     description = "Turn chosen mappings into ordered transformation steps, a template, the recodings still to decide, and the contract a script must follow.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "mappings" => prop("object", "Optional overrides {target_column: source_column}"; additionalProperties = Dict{String, Any}("type" => "string"))],
                            ["source_id", "schema_id"])),
    (name = "run_script", fn = Tools.run_script,
     description = "Evaluate a Julia transformation script against a loaded source, validate the resulting DataFrame against the schema, and optionally write it as SDMx-CSV.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "script" => prop("string", "Julia code that assigns a DataFrame to `result`; `source` is the loaded DataFrame"),
                             "output_path" => prop("string", "Optional path where the result is written as CSV"),
                             "preview_rows" => prop("integer", "Rows of the result to return"; default = 10)],
                            ["source_id", "schema_id", "script"])),
    (name = "validate_csv", fn = Tools.validate_csv,
     description = "Validate an existing SDMx-CSV file against a loaded schema.",
     schema = object_schema(["path" => prop("string", "Path to the CSV file"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "preview_rows" => prop("integer", "Rows to return"; default = 10)],
                            ["path", "schema_id"]))
]

"""
    tool_handler(session, f) -> Function

Wrap a tool function as an MCP handler: stdout is redirected to stderr for the
duration of the call and the result is returned as JSON text.
"""
function tool_handler(session::Tools.Session, f)
    return function (args)
        out = redirect_stdout(stderr) do
            Tools.run_tool(session, f, args)
        end
        return TextContent(text = JSON3.write(out))
    end
end

"""
    mcp_tools(session) -> Vector{MCPTool}
"""
function mcp_tools(session::Tools.Session)
    return [MCPTool(name = spec.name, description = spec.description, input_schema = spec.schema,
                    handler = tool_handler(session, spec.fn), return_type = TextContent)
            for spec in TOOL_SPECS]
end

"""
    serve_mcp(; name="sdmxer-wizard", version=<package version>)

Start the stdio MCP server. The protocol is written to a duplicate of the
original stdout; stdout itself is redirected to stderr so stray prints cannot
corrupt the channel.
"""
function serve_mcp(; name::String="sdmxer-wizard", version::String=string(pkgversion(SDMXerWizard)))
    protocol_fd = Base.Libc.dup(RawFD(1))
    protocol_out = fdio(Int(protocol_fd.fd), true)
    redirect_stdout(stderr)
    global_logger(ConsoleLogger(stderr, Logging.Info))
    session = Tools.Session()
    server = mcp_server(name = name, version = version, tools = mcp_tools(session),
                        description = "SDMXerWizard: map source data to SDMx dataflows and validate the result",
                        instructions = SERVER_INSTRUCTIONS)
    start!(server; transport = StdioTransport(input = stdin, output = protocol_out))
    return nothing
end
```

If `redirect_stdout(stderr)` is refused for the stream type, use `redirect_stdout(Base.stderr)` inside `redirect_stdio`; if `Base.Libc.dup` is not available under that name, use `ccall(:dup, Cint, (Cint,), 1)`.

In `SDMXerWizard.jl`: `include("MCPServer.jl")` after `Tools.jl`, `export serve_mcp, mcp_tools`.

`.mcp.json`:

```json
{
  "mcpServers": {
    "sdmxer-wizard": {
      "command": "julia",
      "args": ["--project=/home/ubuntu/reps/sdmx/julia_sdmx/SDMXerWizard.jl", "--startup-file=no",
               "-e", "using SDMXerWizard; serve_mcp()"]
    }
  }
}
```

- [ ] **Step 4: Run, expect pass.** The stdio round-trip test spawns a second Julia and takes package load time; if `readline` blocks for more than two minutes, run the command by hand and read stderr.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add the MCP server over the Tools session"`

---

### Task 8: Demo, docs, README, start-up time

**Files:**
- Create: `demo/korea/walkthrough.md`, `docs/src/mcp.md`
- Modify: `docs/make.jl`, `README.md`, `Project.toml` (only if a precompile workload is added), `src/SDMXerWizard.jl` (same)

- [ ] **Step 1: Measure cold start**

Run: `time julia --project --startup-file=no -e 'using SDMXerWizard'` twice (second run counts). If wall time is above twenty seconds, add `PrecompileTools` to deps and a `@compile_workload` block at the end of `SDMXerWizard.jl` that builds a `Tools.Session`, registers a schema built from the dry-run fixture code inlined there, loads `demo/korea/inbound_visitors.csv`, runs `infer_mappings` and `transformation_plan`. Re-measure and record both numbers in `docs/src/mcp.md`.

- [ ] **Step 2: Walkthrough**

`demo/korea/walkthrough.md` lists, in order, the prompts to type in Claude Code and the tool call each should trigger:

1. "Load the OECD inbound tourism dataflow: <url>" (load_schema).
2. "Load demo/korea/inbound_visitors.csv" (load_source).
3. "Which columns map to which?" (infer_mappings).
4. "What codes exist for visitor type?" (lookup_codes).
5. "Plan the transformation with Market to REF_AREA, Visitor type to VISITOR_TYPE, Year to TIME_PERIOD, Visitors to OBS_VALUE" (transformation_plan).
6. "Write the script and run it; keep fixing until it validates" (run_script, repeated).
7. "Write the result to out/inbound.csv and validate the file" (run_script with output_path, validate_csv).

With a note on what to say about the guard: the model never sees the full data, only samples, and it never guesses codes without looking them up.

- [ ] **Step 3: Docs page and README**

`docs/src/mcp.md`: how to register the server (the `.mcp.json` snippet), the seven tools in a table, the session model, the script contract verbatim, the sandbox caveat (no timeout, runs in the server process), and the measured start-up time. Add `"MCP tools" => "mcp.md"` to `docs/make.jl` pages. Rewrite README sections "Requirements" (Julia 1.12), "Features", "Philosophy" and "Quick Start" so the first example is the `.mcp.json` registration followed by the pure-Julia use of `Tools`:

```julia
using SDMXerWizard
session = Tools.Session()
schema = Tools.load_schema(session; url = "https://...")
source = Tools.load_source(session; path = "my_data.csv")
Tools.infer_mappings(session; source_id = source["source_id"], schema_id = schema["schema_id"])
```

Remove every mention of Ollama, OpenAI, PromptingTools and `setup_sdmx_llm`.

- [ ] **Step 4: Docs build**

Run: `julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); include("docs/make.jl")'`. Expected: builds without missing-docstring errors (fix any `@docs` block that names a removed function).

- [ ] **Step 5: Full test run and commit**

Run: `julia --project -e 'using Pkg; Pkg.test()'`. Expected: pass. Then `git add -A && git commit -m "Add demo walkthrough, MCP docs and README for the tool surface"`.
