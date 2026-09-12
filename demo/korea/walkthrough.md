# Korea demo walkthrough

One source file, one dataflow, one chat. The server is registered through
the `.mcp.json` at the repository root; open Claude Code in this repository
and the seven tools are available.

The file `inbound_visitors.csv` is a small tourism-style table: source market,
visitor type, year, and visitors in thousands. It stands in for the kind of
spreadsheet a national statistics office receives.

## Prompts, in order

1. **"Load the OECD inbound tourism dataflow from this URL: <url>"**
   Expected call: `load_schema`. Point out the summary: dimensions in order,
   required columns, codelist sizes, and that no codes were returned.

2. **"Load demo/korea/inbound_visitors.csv"**
   Expected call: `load_source`. Point out the five sample values per column:
   the model sees a sample, never the file.

3. **"Which source columns map to which dataflow columns?"**
   Expected call: `infer_mappings`. Show the candidates with confidence and
   the list of required columns that are still unmapped (the constant
   dimensions such as the counterpart area or the measure).

4. **"What codes are there for the visitor type dimension?"**
   Expected call: `lookup_codes`. The model looks codes up instead of
   guessing them.

5. **"Plan the transformation with Market to REF_AREA, Visitor type to
   VISITOR_TYPE, Year to TIME_PERIOD and Visitors to OBS_VALUE"**
   Expected call: `transformation_plan`. Show the recodings section: each
   source label with three candidate codes ranked by similarity, and the
   contract the script must follow.

6. **"Write the script and run it. Keep fixing it until the result validates."**
   Expected calls: `run_script`, possibly several times. The first attempt
   often fails on a type or a missing constant column; the error names the
   line and the model corrects it. This loop is the point of the demo.

7. **"Write the result to out/inbound.csv and validate the file"**
   Expected calls: `run_script` with `output_path`, then `validate_csv`.

## What to say

- The library never calls a model. The model calls the library.
- Nothing the model writes is trusted until the library has run it and
  validated the output against the dataflow structure.
- The same functions work from plain Julia without any model in the loop.

## If the network is down

`load_schema` needs the dataflow URL. For an offline rehearsal, register a
hand-built schema in a Julia session and drive the `Tools` functions directly;
`test/fixtures.jl` shows how to build one.
