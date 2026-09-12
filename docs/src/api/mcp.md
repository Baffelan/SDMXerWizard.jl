# MCP Tools

SDMXerWizard exposes its capabilities to a language model through the Model
Context Protocol. The server holds a session, the model calls tools that read
from and write to that session, and every result is plain JSON.

## Two servers, one flow

Discovery belongs to the [SDMx MCP gateway](https://github.com/Baffelan/sdmx-mcp-gateway):
finding a dataflow by keyword across a provider, inspecting its structure,
browsing or searching the codes of a dimension, checking availability. It
runs as a hosted service, so registering it costs one line. SDMXerWizard
takes over once the dataflow is chosen: it loads the structure with its
codelists, profiles the source, infers mappings, plans the transformation,
runs the model's script and validates the result.

```json
{
  "mcpServers": {
    "sdmx-gateway": {
      "type": "http",
      "url": "https://sdmx-mcp-gateway-production.up.railway.app/mcp"
    },
    "sdmxer-wizard": {
      "command": "julia",
      "args": ["--project=/path/to/SDMXerWizard.jl", "--startup-file=no",
               "-e", "using SDMXerWizard; serve_mcp()"]
    }
  }
}
```

The handoff is by identifier: `load_schema` accepts the `endpoint` key,
`agency` and `dataflow_id` exactly as the gateway's `list_dataflows` reports
them, and builds the structure URL from a provider table that mirrors the
gateway's. A plain `url` still works for providers the table does not know.
The gateway can also be self-hosted from its repository with `uv`; see its
README.

Cold start on a laptop-class machine is about four seconds once the package
is precompiled. Logging and any stray print go to stderr; the protocol channel
is a duplicate of the original stdout.

## The tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `load_schema` | `url`, or `endpoint` + `dataflow_id` (+ `agency`, `version`); `with_codelists` | `schema_id`, dataflow info, dimensions in order, time dimension, attributes, measures, required and optional columns, codelist sizes |
| `load_source` | `path`, `sheet`, `header_row`, `anonymize` | `source_id`, row and column counts, per-column type, missing and unique counts, up to five sample values, suggested key, value and time columns |
| `infer_mappings` | `source_id`, `schema_id`, `method`, `confidence_threshold` | candidates per target column with confidence, match type and evidence; unmapped source, target and required columns; warnings |
| `transformation_plan` | `source_id`, `schema_id`, `mappings` | chosen mappings, ordered steps, loading snippet, template, recodings with candidate codes, the script contract |
| `run_script` | `source_id`, `schema_id`, `script`, `output_path`, `preview_rows` | validation report, preview rows, result size, written path |
| `validate_csv` | `path`, `schema_id`, `preview_rows` | validation report and preview for an existing file |
| `compare_with_published` | `source_id`, `schema_id`, `data_url` or `filters`, `start_period`, `end_period` | rows matched, only on either side, exact and near agreement of OBS_VALUE, a sample of disagreements |

Every tool carries annotations: all are marked read-only except `run_script`,
which is marked destructive because it executes model-written code and may
write a file. Clients that honour the hints approve the read-only tools
silently and show the script before running it.

## The guided prompt

The server publishes one prompt, `map_to_sdmx`, with arguments `file`,
`keywords`, and optional `endpoint` and `output`. Claude Code exposes it as
the slash command `/mcp__sdmxer-wizard__map_to_sdmx`. It walks the model
through discovery on the gateway, loading, profiling, mapping, code lookup,
planning, writing and running the script until it validates, and the
comparison with published data.

## Session and handles

`load_schema` and `load_source` return handles (`schema_1`, `source_2`) that
later tools take as arguments. Code lookups are not needed here: the gateway's
`get_dimension_codes` browses codes, and `transformation_plan` returns ranked
candidate codes for every source value that still needs recoding. The session keeps the parsed schema with its
codelists, the source DataFrame with its profile, and the last mapping result
for each source, so nothing is fetched twice and large objects never pass
through the model's context. Handles live as long as the server process.

## The script contract

`transformation_plan` returns this text under `contract`, and `run_script`
repeats it in the hint of every script failure:

> The script is evaluated in a fresh module where `source` is the loaded
> source DataFrame and the packages DataFrames, CSV and Dates are already in
> scope, together with SDMXer. Do not read the file again. The script must
> assign a DataFrame to a variable named `result` whose columns are the SDMx
> columns of the target schema: dimensions in order, TIME_PERIOD, OBS_VALUE,
> then attributes. Code values must be codelist ids, never labels. Use string
> concatenation rather than string interpolation.

## The sandbox

`run_script` evaluates the script inside a fresh anonymous module with a copy
of the source DataFrame. Parse errors and runtime errors come back as an error
dict naming the failing line. There is no timeout and no resource limit: the
script runs in the server process, on the machine where the server was
started, with that user's permissions. Run the server on your own machine
with scripts you are willing to execute.

## Errors

Every tool result is either the documented shape or a dict with `error` and
`hint`. An unknown handle lists the handles that exist; an unknown method
lists the accepted values; a script failure repeats the contract. The MCP
call itself always succeeds, so the model reads the error and retries.

## Reference

```@docs
SDMXerWizard.Tools
SDMXerWizard.Tools.Session
SDMXerWizard.Tools.ToolError
SDMXerWizard.Tools.run_tool
SDMXerWizard.Tools.load_schema
SDMXerWizard.Tools.register_schema!
SDMXerWizard.Tools.structure_url
SDMXerWizard.Tools.load_source
SDMXerWizard.Tools.infer_mappings
SDMXerWizard.Tools.transformation_plan
SDMXerWizard.Tools.run_script
SDMXerWizard.Tools.validate_csv
SDMXerWizard.Tools.compare_with_published
SDMXerWizard.Tools.compare_frames
SDMXerWizard.mcp_tools
SDMXerWizard.mcp_prompts
SDMXerWizard.serve_mcp
```
