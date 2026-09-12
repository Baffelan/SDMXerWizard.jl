# Korea demo walkthrough

One source file, one dataflow, one chat, two servers. The `.mcp.json` at the
repository root registers the hosted SDMx MCP gateway for discovery and this
package for the transformation; open Claude Code in this repository and both
are available.

The file `inbound_visitors.csv` is a small tourism-style table: source market,
visitor type, year, and visitors in thousands. It stands in for the kind of
spreadsheet a national statistics office receives.

## The short version

Type `/mcp__sdmxer-wizard__map_to_sdmx` with the file path and the keywords
"inbound tourism", and the model runs every step below on its own, pausing
only if several dataflows match. Use the long version when you want to
narrate each call.

## Prompts, in order

0. **"Find the OECD dataflow about inbound tourism"**
   Expected call: `list_dataflows` on the gateway with endpoint OECD. Among
   the tourism dataflows the one to pick is agency `OECD.CFE.TOU`, id
   `DSD_TOURISM_INTER@DF_INBOUND`, named "Inbound tourism". Point out that
   the model searched 1,500 OECD dataflows and received a few hundred bytes.

1. **"Load that dataflow into the wizard"**
   Expected call: `load_schema` with `endpoint`, `agency` and `dataflow_id`
   taken from the gateway result. Point out the summary: dimensions in
   order, required columns, codelist sizes, and that no codes were returned.

2. **"Load demo/korea/inbound_visitors.csv"**
   Expected call: `load_source`. Point out the five sample values per column:
   the model sees a sample, never the file.

3. **"Which source columns map to which dataflow columns?"**
   Expected call: `infer_mappings`. Show the candidates with confidence and
   the list of required columns that are still unmapped (the constant
   dimensions such as the measure or the unit).

4. **"What codes are there for the visitor type dimension?"**
   Expected call: `get_dimension_codes` on the gateway. The model looks codes
   up instead of guessing them, and the lookup costs nothing on our side.

5. **"Plan the transformation with Market to the counterpart area, Visitor
   type to the visitor type dimension, Year to TIME_PERIOD and Visitors to
   OBS_VALUE"**
   Expected call: `transformation_plan`. Show the recodings section: each
   source label with three candidate codes ranked by similarity, and the
   contract the script must follow.

6. **"Write the script and run it. Keep fixing it until the result validates."**
   Expected calls: `run_script`, possibly several times. The first attempt
   often fails on a type or a missing constant column; the error names the
   line and the model corrects it. This loop is the point of the demo.

7. **"Write the result to out/inbound.csv and validate the file"**
   Expected calls: `run_script` with `output_path`, then `validate_csv`.

8. **"Compare it with what OECD publishes for Korea in 2015 and 2016"**
   Expected call: `compare_with_published` with filters on the reference
   area and the two periods, or with a data URL from the gateway's
   `build_data_url`. The demo file is synthetic, so expect disagreements;
   the point is that the tool reports coverage and agreement row by row,
   and with a real file it would confirm the series.

## What to say

- Neither library calls a model. The model calls the libraries.
- Discovery and transformation are separate concerns and separate servers:
  the gateway is shared and hosted, the wizard runs on the presenter's laptop
  next to the data.
- Nothing the model writes is trusted until the wizard has run it and
  validated the output against the dataflow structure.
- The same functions work from plain Julia without any model in the loop.

## If the network is down

Both `list_dataflows` and `load_schema` need the provider online. For an
offline rehearsal, register a hand-built schema in a Julia session and drive
the `Tools` functions directly; `test/fixtures.jl` shows how to build one.
