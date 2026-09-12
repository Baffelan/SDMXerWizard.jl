# SDMXerWizard as a library used by an LLM: MCP tool surface

Date: 2026-09-12. Branch: `feature/mcp-tools`. Status: approved design.

## Goal

Turn SDMXerWizard into a library that a language model uses, instead of a
library that calls a language model. The package exposes its deterministic
capabilities (schema extraction, source profiling, mapping inference,
transformation planning, script execution and SDMx validation) as tools over
the Model Context Protocol (MCP). The calling model, Claude Code or Claude
Desktop for the demo, orchestrates the workflow, writes the transformation
script, and decides recodings; the library supplies the facts and checks the
result.

The first use is a live demo at the Korea workshop days: one Korean source
file mapped to one OECD or PDH dataflow, through chat, ending with a validated
SDMx-CSV.

Constraints agreed with the author:

- Lives on the branch `feature/mcp-tools`. No backward compatibility with the
  0.1.x API is required.
- Julia 1.12 only.
- Internal LLM code is removed rather than kept alongside.
- Single-dataflow path only. Cross-dataflow joins, MCP resources and prompts,
  and the HTTP transport are out of scope.

## Architecture

Three layers, each usable without the one above it:

1. **Core** (existing, trimmed): data sources, profiling, anonymisation,
   mapping inference, script templates and the transformation step builder. Pure Julia, returns Julia structs.
2. **`SDMXerWizard.Tools`** (new, `src/Tools.jl`): a `Session` and seven
   functions that take a session plus plain arguments and return
   `Dict{String,Any}` values containing only strings, numbers, booleans,
   `nothing`, vectors and nested dicts. No MCP dependency. This is the unit
   the tests exercise.
3. **MCP server** (new, `src/MCPServer.jl`): builds one `MCPTool` per tool
   function with an explicit JSON input schema, and `serve_mcp()` starts a
   stdio server over a global session. Uses ModelContextProtocol.jl as a hard
   dependency.

## Session

```julia
mutable struct Session
    schemas::Dict{String, SchemaEntry}   # "schema_1" => ...
    sources::Dict{String, SourceEntry}   # "source_1" => ...
    counter::Int
end
struct SchemaEntry
    schema::DataflowSchema
    codelists::Union{DataFrame, Nothing}
    origin::String                        # URL or identifier given
end
mutable struct SourceEntry
    data::DataFrame
    profile::SourceDataProfile
    path::String
    last_mapping::Union{AdvancedMappingResult, Nothing}
end
```

Handles are `schema_<n>` and `source_<n>` with a shared counter. One global
session is created inside `serve_mcp()`; tools take the session as first
argument so tests build their own. An unknown handle produces an error result
that lists the handles that exist.

## Tool surface

Every tool function has the signature `tool_name(session::Session; kwargs...)`
and a docstring that becomes the MCP description. Argument names below are
the MCP argument names.

### `load_schema`

Arguments: `url` (string; a dataflow URL as given by the .Stat Data Explorer,
or the path of a local SDMx-ML structure file, both handled by SDMXer's
`extract_dataflow_schema`), `with_codelists` (bool, default true).

Returns: `schema_id`, `dataflow` (id, agency, version, name), `dimensions`
(ordered list of id, position, codelist id, whether time), `time_dimension`,
`attributes` (id, assignment status, relationship, codelist id), `measures`,
`required_columns`, `optional_columns`, `codelist_count`, and per codelist the
number of codes. Codes themselves are never returned here.

### `lookup_codes`

Arguments: `schema_id`, `dimension` (dimension or attribute id), `query`
(optional string matched case-insensitively against code id and name),
`limit` (int, default 20).

Returns: `dimension`, `codelist_id`, `total_codes`, `matches` (list of code id,
name, parent id), `truncated` (bool). With no query the first `limit` codes are
returned. Matching is substring first, then the package's `fuzzy_match_score`
ranking, so "overnight" finds "Overnight visitors" style names.

### `load_source`

Arguments: `path` (string; CSV or Excel), `sheet` (int or string, default 1),
`header_row` (int, default 1), `anonymize` (bool, default false).

Returns: `source_id`, `path`, `file_type`, `row_count`, `column_count`,
`columns` (name, type name, missing count, unique count, up to five sample
values as strings truncated to 60 characters, is categorical, is temporal,
temporal format), `suggested_key_columns`, `suggested_value_columns`,
`suggested_time_columns`, `data_quality_score`. When `anonymize` is true the
stored DataFrame is the anonymised one from `anonymize_source_data`, so nothing
downstream sees the original values.

### `infer_mappings`

Arguments: `source_id`, `schema_id`, `method` (enum heuristic, fuzzy,
advanced; default advanced), `confidence_threshold` (number, default 0.3).

Runs `infer_advanced_mappings` (or the heuristic path for `heuristic`) with an
engine built from the threshold, stores the result on the source entry, and
returns `mappings` grouped by target column with ranked candidates (source
column, confidence, confidence level, match type, evidence, suggested
transformation, notes), `unmapped_source_columns`,
`unmapped_target_columns`, `unmapped_required_columns`, `quality_score`,
`recommendations`, and `warnings` from `validate_mapping_quality(result,
schema)`.

### `transformation_plan`

Arguments: `source_id`, `schema_id`, `mappings` (optional object
`{target_column: source_column}` overriding the inferred candidates).

Uses the stored mapping result (running `infer_mappings` with defaults if none
exists), applies overrides, and returns `steps` from
`build_transformation_steps` (name, operation type, description, source and
target columns, transformation logic, validation checks), `loading_code` from
the loading-code helper, `template` (name and skeleton of the selected
template), `recodings` (for each mapped dimension with a codelist: the distinct
source values that do not already equal a code id, each with up to three
candidate codes from the codelist ranked by fuzzy score), and `contract`, a
short text stating what `run_script` expects.

The contract: the script runs in a module where `source` is the loaded
DataFrame and `DataFrames`, `CSV`, `Dates` and `SDMXer` are available; it must
assign a DataFrame to a variable named `result` whose columns are the SDMx
columns of the target schema.

### `run_script`

Arguments: `source_id`, `schema_id`, `script` (string of Julia code),
`output_path` (optional string), `preview_rows` (int, default 10).

Evaluates the script in a fresh anonymous module per call, with `source` bound
to a copy of the source DataFrame. On success: builds a validator with
`create_validator(schema)`, runs `validate_sdmx_csv`, writes `result` to
`output_path` as CSV if given, and returns `validation` (see below),
`preview` (first rows as row dicts), `result_rows`, `result_columns`, and
`output_path`. On any exception (parse error, runtime error, `result` missing
or not a DataFrame) returns the error result with the message and, when
available, the line number.

### `validate_csv`

Arguments: `path`, `schema_id`, `preview_rows` (int, default 10).

Reads the CSV and returns the same `validation` and `preview` as `run_script`.

### Validation result shape

`compliance_status`, `overall_score`, `total_rows`, `total_columns`, `issues`
(rule id, severity, message, location, number of affected rows and up to ten
affected row indices, suggested fix), `recommendations`, `statistics`.

## Serialisation

All conversions live in `src/Tools.jl` as methods of an internal `to_json`
function: `SourceDataProfile`, `ColumnProfile`, `MappingCandidate`,
`AdvancedMappingResult`, `TransformationStep`, `ScriptTemplate`, and SDMXer's
`ValidationResult` and `ValidationIssue`. Rules:

- Julia `Type` values become their name as a string.
- `missing` becomes `nothing`.
- Symbols and enums become strings.
- DataFrames become vectors of row dicts, capped at the caller's `limit`.
- Sample values are capped at five per column and 60 characters each.
- Numeric stats are rounded to four significant digits.

The `Dict`s pass through `JSON3.write` unchanged; the MCP layer wraps the JSON
text in a `TextContent`.

## Error handling

Each MCP handler calls the tool inside a `try`. A thrown exception becomes
`Dict("error" => message, "hint" => hint)` where the hint is tool-specific:
existing handles for a bad handle, the accepted enum values for a bad
`method`, the contract text for a script failure. The MCP call itself always
succeeds, so the model reads the error and retries.

Tools never write to stdout. Logging goes through `@info` and `@warn` to
stderr. `serve_mcp()` sets the global logger to a stderr logger before
starting. Because SDMXer itself prints some errors with `println`, every MCP
handler runs the tool body inside `redirect_stdout(stderr) do ... end`, so
stray prints reach stderr and the stdio protocol channel stays clean. The
tests cover this with a tool body that prints and a check that the handler's
returned content is still valid JSON.

## Script sandbox

`run_script` uses `Module()` to create an anonymous module, evaluates
`using DataFrames, CSV, Dates, SDMXer` inside it, assigns `source`, parses the
script with `Meta.parseall`, and evaluates it. No timeout and no resource
limits on this branch: the server runs on the presenter's laptop with the
presenter's own scripts. This is stated in the docs page.

## Removals

From `Project.toml`: `PromptingTools`, `GoogleGenAI`, `Llama`, `OpenAI` and
their extension entries and compat lines. Added: `ModelContextProtocol` (hard
dependency), `StructTypes` only if needed by serialisation (expected not).
Compat `julia = "1.12"`. Version 0.2.0.

Source files removed: `SDMXPromptingIntegration.jl`, `SDMXPromptGeneration.jl`,
`SDMXCrossDataflowLLM.jl`, `SDMXJoinPrompts.jl`, `SDMXLLMPipelineOps.jl`,
`SDMXEnhancedTransformation.jl`, `SDMXWorkflow.jl`, `SDMXJoinWorkflow.jl` (it
calls the LLM join script generator) and `SDMXMetadataContext.jl` (it exists
only to build prompt context and calls an undefined Excel analysis function).

`SDMXScriptGeneration.jl` is trimmed: `ScriptGenerator`, `create_script_generator`,
`generate_transformation_script`, `generate_transformation_script_text`,
`create_comprehensive_script_prompt`, `create_script_guidance` and
`post_process_generated_code` go; `TransformationStep`, `ScriptTemplate`,
`GeneratedScript`, the four template constructors, `select_template` (rewritten
to take a profile instead of a generator), `build_transformation_steps`,
`get_loading_code`, `generate_transformation_logic`, `create_validation_logic`,
`calculate_script_complexity`, `validate_generated_script` and
`preview_script_output` stay.

`ext/SDMXerWizardOpenAIExt.jl` and `ext/SDMXerWizardGoogleGenAIExt.jl` are
removed. `ext/SDMXerWizardPlotsExt.jl` stays if it does not reference removed
code.

The module docstring, exports and `LLMProvider`/`ScriptStyle` enums are
rewritten or removed to match.

## Files added

- `src/Tools.jl`: `module Tools`, `Session`, entries, seven tool functions,
  `to_json` methods, `sandbox_run`.
- `src/MCPServer.jl`: `mcp_tools(session)` returning `Vector{MCPTool}`,
  `serve_mcp(; name, version)`.
- `.mcp.json`: Claude Code registration, command
  `julia --project=<repo> --startup-file=no -e "using SDMXerWizard; serve_mcp()"`.
- `demo/korea/inbound_visitors.csv`: small tourism-style source file with
  market, visitor type, year and a value column, in the shape of the dry-run
  test data.
- `demo/korea/walkthrough.md`: the prompts to type on stage and what each
  tool call should show.
- `docs/src/mcp.md`: the tool surface, the session model, the script contract,
  and the sandbox caveat.
- `test/test_tools.jl`: offline tests for the Tools module and the MCP wiring.

## Testing

`test/test_tools.jl` builds a `Session`, registers the hand-built offline
`DataflowSchema` from `test_korea_dryrun.jl` directly into it (a helper
`register_schema!(session, schema, codelists)` exists for this and for
programmatic use), and:

- loads the demo CSV through `load_source` and checks the result is
  JSON-serialisable (round trip through `JSON3.write` and `JSON3.read`) and
  carries the expected columns;
- runs `infer_mappings` with each method and checks `VISITOR_TYPE` is offered;
- calls `transformation_plan` with and without overrides and checks steps and
  the contract are present;
- calls `run_script` with a correct script and checks `compliance_status` and
  the preview, then with a script that throws and with one that does not
  define `result`, checking the error dict each time;
- writes a CSV and checks `validate_csv` agrees with `run_script`;
- checks a bad handle returns the error dict listing existing handles;
- for the MCP layer, calls `mcp_tools(session)`, checks seven tools with the
  expected names and required arguments, and invokes one handler with a `Dict`
  to confirm it returns `TextContent` holding JSON.

`test_korea_dryrun.jl` keeps the tests that still apply. Tests referencing
removed functions in `simple_test.jl` and `test_cross_dataflow.jl` are removed
or trimmed. Aqua stays in the suite.

## Launch and performance

Cold start of `using SDMXerWizard; serve_mcp()` is measured on the
development machine. If it exceeds about twenty seconds, a PrecompileTools
workload covering `load_source`, `infer_mappings` and `run_script` on the demo
CSV is added in the same branch.

## Documentation

`docs/src/llm.md`, `prompts.md`, `workflow.md` and `crossdataflow.md` are
removed from `make.jl` and deleted; `mcp.md` is added. The README requirements,
features and quick start sections are rewritten for the new philosophy: the
library gives the model eyes and a checker, and the model writes the code.

## Amendment, 2026-09-12: discovery delegated to the SDMx MCP gateway

Approved after the first implementation. The hosted
[SDMx MCP gateway](https://github.com/Baffelan/sdmx-mcp-gateway) is
registered alongside this server in `.mcp.json` (HTTP transport, nothing to
install). It owns dataflow discovery, structure inspection, code browsing and
availability. Consequences for this package:

- `lookup_codes` is removed; the gateway's `get_dimension_codes` replaces it
  and `transformation_plan` still returns candidate codes for recodings.
- `load_schema` accepts `endpoint`, `dataflow_id`, `agency` and `version` as
  reported by the gateway, in addition to `url`. A provider table in `Tools`
  (`PROVIDERS`, keyed like the gateway's endpoints) builds the structure URL.
- The server instructions send discovery to the gateway and keep loading,
  mapping, planning, running and validating here.
- Six tools remain: `load_schema`, `load_source`, `infer_mappings`,
  `transformation_plan`, `run_script`, `validate_csv`.

## Amendment, 2026-09-12: prompt, annotations, cross-check, skill

- A `map_to_sdmx` MCP prompt guides the whole flow; Claude Code exposes it as
  a slash command.
- Tool annotations: every tool read-only except `run_script`, marked
  destructive; `load_schema` and `compare_with_published` are open-world.
- A seventh tool, `compare_with_published`, fetches the provider's published
  data (by `data_url` from the gateway or by `filters` through SDMXer's
  `construct_data_url`) and compares it with the last `run_script` result
  on the shared dimension and time columns.
- `SourceEntry` keeps `last_result` for that comparison.
- A project skill at `.claude/skills/sdmx-mapping/SKILL.md` gives Claude Code
  the two-server flow, the contract and recurring validation pitfalls.
- Elicitation before `run_script` was considered and not built: the client's
  own approval dialog, driven by the destructive annotation, already shows
  the script before it runs, and the server-initiated elicitation path is
  not yet established for the clients in use.
