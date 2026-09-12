---
name: sdmx-mapping
description: Use when mapping a CSV or Excel file to an SDMx dataflow with the sdmx-gateway and sdmxer-wizard MCP servers, writing the transformation script, or fixing a validation report from run_script.
---

# Mapping a file to an SDMx dataflow

Two MCP servers share the work. Use each for what it is for.

| Need | Server and tool |
| --- | --- |
| Find a dataflow by keyword | `sdmx-gateway` `list_dataflows` |
| Inspect a structure, browse or search codes | `sdmx-gateway` `get_dataflow_structure`, `get_dimension_codes` |
| Check what data exists, build a data URL | `sdmx-gateway` `get_data_availability`, `build_data_url` |
| Load the structure with codelists | `sdmxer-wizard` `load_schema` (endpoint, agency, dataflow_id from the gateway) |
| Profile the file, infer mappings, plan | `sdmxer-wizard` `load_source`, `infer_mappings`, `transformation_plan` |
| Run and validate the script | `sdmxer-wizard` `run_script`, `validate_csv` |
| Compare with published data | `sdmxer-wizard` `compare_with_published` |

The `/mcp__sdmxer-wizard__map_to_sdmx` prompt runs the whole sequence.

## Rules

- Never invent a code. Look it up with `get_dimension_codes`, or take it from
  the `recodings` section of `transformation_plan`.
- Handles (`schema_1`, `source_1`) belong to the running wizard server and
  vanish when it restarts. Reload rather than guess.
- The model never sees the file, only five sample values per column. Do not
  ask for more rows; work from the profile and the recodings.

## The script contract

`run_script` evaluates the script in a fresh module where `source` is the
loaded DataFrame and DataFrames, CSV, Dates and SDMXer are in scope. The
script must assign a DataFrame to `result` with the schema's columns:
dimensions in order, TIME_PERIOD, OBS_VALUE, then attributes. Use string
concatenation, never string interpolation.

## Recurring validation failures

- `TIME_PERIOD` must be text (`string.(source[!, "Year"])`), never an integer.
- Every dimension column is required, including the constant ones (measure,
  unit, source). Fill them with `fill("CODE", nrow(source))`.
- Code values are codelist ids, never labels: `"OVN"`, not `"Overnight visitors"`.
- `OBS_VALUE` must be numeric; convert with `Float64.(...)` and apply unit
  multipliers explicitly.
- A missing lookup key throws `KeyError` at the line reported; add the
  missing label to the recoding dictionary rather than dropping the row.
