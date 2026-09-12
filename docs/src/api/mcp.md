# MCP Tools

SDMXerWizard exposes its capabilities to a language model through the Model
Context Protocol. The server holds a session, the model calls tools that read
from and write to that session, and every result is plain JSON.

## Registering the server

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

Cold start on a laptop-class machine is about four seconds once the package
is precompiled. Logging and any stray print go to stderr; the protocol channel
is a duplicate of the original stdout.

## The tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `load_schema` | `url`, `with_codelists` | `schema_id`, dataflow info, dimensions in order, time dimension, attributes, measures, required and optional columns, codelist sizes |
| `lookup_codes` | `schema_id`, `dimension`, `query`, `limit` | matching codes with names and parents, total count, whether the list was truncated |
| `load_source` | `path`, `sheet`, `header_row`, `anonymize` | `source_id`, row and column counts, per-column type, missing and unique counts, up to five sample values, suggested key, value and time columns |
| `infer_mappings` | `source_id`, `schema_id`, `method`, `confidence_threshold` | candidates per target column with confidence, match type and evidence; unmapped source, target and required columns; warnings |
| `transformation_plan` | `source_id`, `schema_id`, `mappings` | chosen mappings, ordered steps, loading snippet, template, recodings with candidate codes, the script contract |
| `run_script` | `source_id`, `schema_id`, `script`, `output_path`, `preview_rows` | validation report, preview rows, result size, written path |
| `validate_csv` | `path`, `schema_id`, `preview_rows` | validation report and preview for an existing file |

## Session and handles

`load_schema` and `load_source` return handles (`schema_1`, `source_2`) that
later tools take as arguments. The session keeps the parsed schema with its
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
SDMXerWizard.Tools.lookup_codes
SDMXerWizard.Tools.load_source
SDMXerWizard.Tools.infer_mappings
SDMXerWizard.Tools.transformation_plan
SDMXerWizard.Tools.run_script
SDMXerWizard.Tools.validate_csv
SDMXerWizard.mcp_tools
SDMXerWizard.serve_mcp
```
