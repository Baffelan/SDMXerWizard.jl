"""
MCP server over a `Tools.Session`.

`mcp_tools(session)` turns the seven tool functions into `MCPTool` values
with explicit JSON input schemas; `serve_mcp()` starts a stdio server for
Claude Code, Claude Desktop or any other MCP client.
"""

using ModelContextProtocol

const SERVER_INSTRUCTIONS = """
SDMXerWizard maps a source data file (CSV or Excel) to an SDMx dataflow and validates the result. \
Discovery is the job of the sdmx-gateway server when it is available: use its list_dataflows to find \
the dataflow, get_dataflow_structure to inspect it, and get_dimension_codes to browse or search codes. \
Then, here: load_schema (with the endpoint, agency and dataflow id the gateway reported, or a URL), \
load_source, infer_mappings, transformation_plan, and finally write a Julia script that follows the \
returned contract and submit it with run_script. Iterate on the script until the validation report is \
compliant. Handles such as schema_1 and source_1 refer to objects held in this server's session; they \
do not survive a restart."""

function prop(type::String, description::String; extra...)
    d = Dict{String, Any}("type" => type, "description" => description)
    for (k, v) in pairs(extra)
        d[string(k)] = v
    end
    return d
end

function object_schema(props::Vector{Pair{String, Dict{String, Any}}}, required::Vector{String})
    return Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}(props...),
                             "required" => required)
end

# Behavioural hints clients use for approval decisions. Read-only tools only
# touch the session; run_script executes model-written code and may write a file.
function hints(; read_only::Bool, network::Bool=false, destructive::Bool=false)
    return Dict{String, Any}("readOnlyHint" => read_only, "destructiveHint" => destructive,
                             "idempotentHint" => read_only, "openWorldHint" => network)
end

const TOOL_SPECS = [
    (name = "load_schema", fn = Tools.load_schema, annotations = hints(read_only = true, network = true),
     description = "Fetch an SDMx dataflow structure and store it in the session. Give either url, or endpoint plus dataflow_id (and agency for providers such as OECD that publish under sub-agencies) as reported by the sdmx-gateway tools. Returns schema_id, dimensions in order, attributes, measures, required columns and codelist sizes.",
     schema = object_schema(["url" => prop("string", "Dataflow structure URL (for example a .Stat Data Explorer developer API link with references=all) or path to a structure XML file"),
                             "endpoint" => prop("string", "Provider key as used by the sdmx-gateway server"; enum = sort(collect(keys(Tools.PROVIDERS)))),
                             "dataflow_id" => prop("string", "Dataflow id on that provider, for example DF_BP50 or DSD_TOURISM@DF_INBOUND"),
                             "agency" => prop("string", "Agency owning the dataflow; defaults to the provider's own agency"),
                             "version" => prop("string", "Dataflow version"; default = "latest"),
                             "with_codelists" => prop("boolean", "Also fetch the codelists so value matching and recodings work"; default = true)],
                            String[])),
    (name = "load_source", fn = Tools.load_source, annotations = hints(read_only = true),
     description = "Read a CSV or Excel file, profile its columns and store it in the session. Returns source_id and a compact column profile with a few sample values per column.",
     schema = object_schema(["path" => prop("string", "Absolute path to a .csv, .xlsx or .xls file"),
                             "sheet" => prop("integer", "Excel sheet number"; default = 1),
                             "header_row" => prop("integer", "Row holding the column names"; default = 1),
                             "anonymize" => prop("boolean", "Replace values with synthetic ones before anything is returned"; default = false)],
                            ["path"])),
    (name = "infer_mappings", fn = Tools.infer_mappings, annotations = hints(read_only = true),
     description = "Rank source columns against the target columns of a schema and report unmapped required columns.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "method" => prop("string", "heuristic (names only), fuzzy (names and statistics) or advanced (adds codelist value matching)"; enum = ["heuristic", "fuzzy", "advanced"], default = "advanced"),
                             "confidence_threshold" => prop("number", "Drop candidates below this confidence"; default = 0.3)],
                            ["source_id", "schema_id"])),
    (name = "transformation_plan", fn = Tools.transformation_plan, annotations = hints(read_only = true),
     description = "Turn chosen mappings into ordered transformation steps, a template, the recodings still to decide, and the contract a script must follow.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "mappings" => prop("object", "Optional overrides {target_column: source_column}"; additionalProperties = Dict{String, Any}("type" => "string"))],
                            ["source_id", "schema_id"])),
    (name = "run_script", fn = Tools.run_script, annotations = hints(read_only = false, destructive = true),
     description = "Evaluate a Julia transformation script against a loaded source, validate the resulting DataFrame against the schema, and optionally write it as SDMx-CSV. Errors come back with the failing line.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "script" => prop("string", "Julia code that assigns a DataFrame to `result`; `source` is the loaded DataFrame"),
                             "output_path" => prop("string", "Optional path where the result is written as CSV"),
                             "preview_rows" => prop("integer", "Rows of the result to return"; default = 10)],
                            ["source_id", "schema_id", "script"])),
    (name = "validate_csv", fn = Tools.validate_csv, annotations = hints(read_only = true),
     description = "Validate an existing SDMx-CSV file against a loaded schema.",
     schema = object_schema(["path" => prop("string", "Path to the CSV file"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "preview_rows" => prop("integer", "Rows to return"; default = 10)],
                            ["path", "schema_id"])),
    (name = "compare_with_published", fn = Tools.compare_with_published, annotations = hints(read_only = true, network = true),
     description = "Fetch what the provider already publishes for the dataflow and compare it with the last run_script result on the shared dimension and time columns: coverage on both sides and OBS_VALUE agreement. Give data_url from the sdmx-gateway build_data_url tool, or filters such as {\"REF_AREA\": \"KOR\"} and the URL is built from the schema.",
     schema = object_schema(["source_id" => prop("string", "Handle returned by load_source"),
                             "schema_id" => prop("string", "Handle returned by load_schema"),
                             "data_url" => prop("string", "Complete SDMx data query URL, for example from the gateway's build_data_url"),
                             "filters" => prop("object", "Dimension filters {dimension_id: code} used to build the data URL when data_url is absent"; additionalProperties = Dict{String, Any}("type" => "string")),
                             "start_period" => prop("string", "First period to fetch, for example 2015"),
                             "end_period" => prop("string", "Last period to fetch, for example 2016")],
                            ["source_id", "schema_id"]))
]

"""
    tool_handler(session, f) -> Function

Wrap a tool function as an MCP handler. Stdout is redirected to stderr for
the duration of the call, so a stray print inside the library cannot reach
the protocol channel, and the result is returned as JSON text.
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

The seven SDMXerWizard tools bound to `session`.
"""
function mcp_tools(session::Tools.Session)
    return MCPTool[MCPTool(name = spec.name, description = spec.description, input_schema = spec.schema,
                           handler = tool_handler(session, spec.fn), return_type = TextContent,
                           annotations = spec.annotations)
                   for spec in TOOL_SPECS]
end

const MAP_TO_SDMX_TEXT = """
Map the source file {file} to an SDMx dataflow and produce a validated SDMx-CSV. Work through these steps \
and report after each one.

1. Find the dataflow: call list_dataflows on the sdmx-gateway server with the keywords {keywords}{?endpoint? on endpoint {endpoint}}. \
If several dataflows fit, list them and ask me which one before going on.
2. Load it here: call load_schema with the endpoint, agency and dataflow_id the gateway reported.
3. Load the file: call load_source with {file}.
4. Call infer_mappings and read the candidates and the unmapped required columns.
5. Where a mapping or a code is unsure, call get_dimension_codes on the sdmx-gateway server. Never invent a code.
6. Call transformation_plan with the mappings you chose, and read the recodings and the contract.
7. Write a Julia script that follows the contract and submit it with run_script. Read the validation report and \
fix the script until compliance_status is compliant{?output?, then run it once more with output_path {output}}.
8. If the provider publishes this dataflow for the same area, call compare_with_published and report the agreement.
9. Summarise what was mapped, what was recoded, and what validation said."""

"""
    mcp_prompts() -> Vector{MCPPrompt}

The guided `map_to_sdmx` prompt, which clients expose as a slash command.
"""
function mcp_prompts()
    return MCPPrompt[MCPPrompt(
        name = "map_to_sdmx",
        title = "Map a file to an SDMx dataflow",
        description = "Guide the whole flow: find the dataflow with the gateway, load it, profile the file, infer mappings, plan, write and run the script until it validates.",
        arguments = [PromptArgument(name = "file", description = "Path to the CSV or Excel file", required = true),
                     PromptArgument(name = "keywords", description = "Words that identify the dataflow, for example: inbound tourism", required = true),
                     PromptArgument(name = "endpoint", description = "Provider key, for example OECD or SPC", required = false),
                     PromptArgument(name = "output", description = "Path where the validated SDMx-CSV should be written", required = false)],
        messages = [PromptMessage(content = TextContent(text = MAP_TO_SDMX_TEXT))])]
end

"""
    serve_mcp(; name="sdmxer-wizard", version=<package version>)

Start the stdio MCP server over a fresh session and block until the client
disconnects. The protocol is written to a duplicate of the original stdout;
stdout itself is redirected to stderr so stray prints cannot corrupt the
channel.
"""
function serve_mcp(; name::String="sdmxer-wizard", version::String=string(pkgversion(SDMXerWizard)))
    protocol_fd = Base.Libc.dup(RawFD(1))
    protocol_out = fdio(Base.cconvert(Cint, protocol_fd), true)
    redirect_stdout(stderr)
    global_logger(ConsoleLogger(stderr, Logging.Info))
    session = Tools.Session()
    server = mcp_server(name = name, version = version, tools = mcp_tools(session), prompts = mcp_prompts(),
                        description = "SDMXerWizard: map source data to SDMx dataflows and validate the result",
                        instructions = SERVER_INSTRUCTIONS)
    start!(server; transport = StdioTransport(input = stdin, output = protocol_out))
    return nothing
end
