using Test, JSON3, DataFrames, CSV
using ModelContextProtocol
using SDMXer, SDMXerWizard
using SDMXerWizard.Tools
@isdefined(fixture_schema) || include("fixtures.jl")

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
    @test anon["anonymized"] == true
    @test !("China" in string.(Tools.get_source(session, "source_2").data[!, "Market"]))

    missing_handle = Tools.run_tool(session, (s; source_id) -> Tools.get_source(s, source_id), Dict("source_id" => "source_9"))
    @test occursin("source_1", missing_handle["hint"])
end

@testset "Schema registration and structure URLs" begin
    session = Tools.Session()
    id = Tools.register_schema!(session, fixture_schema(), fixture_codelists(); origin="fixture")
    @test id == "schema_1"
    summary = Tools.schema_summary(session.schemas[id], id)
    @test summary["schema_id"] == id
    @test [d["id"] for d in summary["dimensions"]] == ["REF_AREA", "COUNTERPART_AREA", "VISITOR_TYPE", "MEASURE"]
    @test summary["time_dimension"]["id"] == "TIME_PERIOD"
    @test "OBS_VALUE" in summary["required_columns"]
    @test summary["codelists"]["CL_AREA"] == 3
    @test summary["codelists_loaded"] == true
    @test roundtrip(summary)["schema_id"] == id
    @test Tools.codelist_for(session.schemas[id], "UNIT_MEASURE") == "CL_UNIT"
    @test Tools.codelist_for(session.schemas[id], "COUNTERPART_AREA") === nothing
    @test nrow(Tools.codes_for(session.schemas[id], "CL_VISITOR_TYPE")) == 2

    # Handoff from the SDMx MCP gateway: endpoint key plus dataflow id
    @test Tools.structure_url("OECD", "DSD_TOURISM@DF_INBOUND"; agency="OECD.CFE.EDS") ==
          "https://sdmx.oecd.org/public/rest/dataflow/OECD.CFE.EDS/DSD_TOURISM@DF_INBOUND/latest?references=all"
    @test Tools.structure_url("spc", "DF_BP50") ==
          "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all"
    @test Tools.structure_url("IMF", "CPI"; version="3.0") ==
          "https://api.imf.org/external/sdmx/2.1/dataflow/IMF.STA/CPI/3.0?references=all"
    @test_throws Tools.ToolError Tools.structure_url("NOPE", "DF_X")

    neither = Tools.run_tool(session, Tools.load_schema, Dict{String, Any}())
    @test occursin("url", neither["error"])
    unknown = Tools.run_tool(session, Tools.load_schema, Dict("endpoint" => "NOPE", "dataflow_id" => "DF_X"))
    @test occursin("SPC", unknown["hint"])
    @test occursin("OECD", unknown["hint"])
end

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
        @test "Visitor type" in [c["source_column"] for c in vt["candidates"]]
        @test vt["candidates"][1]["source_column"] == "Visitor type"
        @test vt["codelist_id"] == "CL_VISITOR_TYPE"
        @test haskey(out, "unmapped_required_columns")
        @test out["warnings"] isa Vector
    end
    @test Tools.get_source(session, sid).last_mapping isa AdvancedMappingResult
    bad = Tools.run_tool(session, Tools.infer_mappings, Dict("source_id" => sid, "schema_id" => schid, "method" => "llm"))
    @test occursin("heuristic", bad["hint"])
end

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
    # infer_mappings ran implicitly
    @test Tools.get_source(session, sid).last_mapping isa AdvancedMappingResult

    overridden = Tools.run_tool(session, Tools.transformation_plan,
        Dict("source_id" => sid, "schema_id" => schid,
             "mappings" => Dict("REF_AREA" => "Market", "VISITOR_TYPE" => "Visitor type",
                                "TIME_PERIOD" => "Year", "OBS_VALUE" => "Visitors ('000)")))
    @test !haskey(overridden, "error")
    chosen_targets = Set(m["target_column"] for m in overridden["chosen_mappings"])
    @test issubset(Set(["REF_AREA", "VISITOR_TYPE", "TIME_PERIOD", "OBS_VALUE"]), chosen_targets)
    @test count(m -> m["target_column"] == "REF_AREA", overridden["chosen_mappings"]) == 1
    rec = only(filter(r -> r["target_column"] == "VISITOR_TYPE", overridden["recodings"]))
    ov = only(filter(v -> v["source_value"] == "Overnight visitors", rec["values"]))
    @test ov["candidates"][1]["code"] == "OVN"
    area = only(filter(r -> r["target_column"] == "REF_AREA", overridden["recodings"]))
    @test any(v -> v["source_value"] == "USA", area["values"])
    @test isempty(filter(r -> r["target_column"] == "TIME_PERIOD", overridden["recodings"]))
    @test !("REF_AREA" in overridden["unmapped_required_columns"])

    bad = Tools.run_tool(session, Tools.transformation_plan,
        Dict("source_id" => sid, "schema_id" => schid, "mappings" => Dict("REF_AREA" => "Nope")))
    @test occursin("Nope", bad["error"])
end

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
        Dict("source_id" => sid, "schema_id" => schid, "script" => GOOD_SCRIPT,
             "output_path" => outpath, "preview_rows" => 3))
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

    notdf = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "result = 42"))
    @test occursin("DataFrame", notdf["error"])

    parsefail = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "result = DataFrame(("))
    @test haskey(parsefail, "error")

    # the sandbox works on a copy: the stored source is untouched
    mutate = Tools.run_tool(session, Tools.run_script,
        Dict("source_id" => sid, "schema_id" => schid, "script" => "select!(source, Not(\"Market\"))\nresult = source"))
    @test !haskey(mutate, "error")
    @test "Market" in names(Tools.get_source(session, sid).data)

    vc = Tools.run_tool(session, Tools.validate_csv, Dict("path" => outpath, "schema_id" => schid))
    @test vc["validation"]["compliance_status"] == ok["validation"]["compliance_status"]
    @test vc["result_rows"] == 10
end

@testset "compare_with_published" begin
    session = Tools.Session()
    sid = Tools.run_tool(session, Tools.load_source, Dict("path" => DEMO_CSV))["source_id"]
    schid = Tools.register_schema!(session, fixture_schema(), fixture_codelists();
        origin="https://example.org/rest/dataflow/TEST/DF_INBOUND/1.0?references=all")
    early = Tools.run_tool(session, Tools.compare_with_published, Dict("source_id" => sid, "schema_id" => schid))
    @test occursin("run_script", early["hint"])

    ok = Tools.run_tool(session, Tools.run_script, Dict("source_id" => sid, "schema_id" => schid, "script" => GOOD_SCRIPT))
    @test !haskey(ok, "error")
    ours = Tools.get_source(session, sid).last_result
    @test ours isa DataFrame

    parts = Tools.parse_structure_url("https://example.org/rest/dataflow/TEST/DF_INBOUND/1.0?references=all")
    @test parts.base_url == "https://example.org/rest"
    @test parts.agency == "TEST" && parts.dataflow_id == "DF_INBOUND" && parts.version == "1.0"
    @test_throws Tools.ToolError Tools.parse_structure_url("/local/structure.xml")

    published = copy(ours)
    published[1, "OBS_VALUE"] = published[1, "OBS_VALUE"] * 1.005   # within one percent
    published[2, "OBS_VALUE"] = published[2, "OBS_VALUE"] * 2.0     # disagreement
    published = published[1:end-1, :]                               # one row only in ours
    push!(published, published[1, :]); published[end, "TIME_PERIOD"] = "2099"  # one row only published
    cmp = Tools.compare_frames(fixture_schema(), ours, published)
    @test "REF_AREA" in cmp["key_columns"] && "TIME_PERIOD" in cmp["key_columns"]
    @test cmp["matched"] == 9
    @test cmp["only_in_ours"] == 1
    @test cmp["only_in_published"] == 1
    @test cmp["exact_agreement"] == 7
    @test cmp["within_one_percent"] == 1
    @test cmp["disagreements"] == 1
    @test length(cmp["disagreement_sample"]) == 1
    @test roundtrip(cmp)["matched"] == 9

    @test_throws Tools.ToolError Tools.compare_frames(fixture_schema(), ours, DataFrame(FOO = [1]))
end

@testset "MCP wiring" begin
    session = Tools.Session()
    tools = SDMXerWizard.mcp_tools(session)
    @test Set(t.name for t in tools) == Set(["load_schema", "load_source", "infer_mappings",
                                             "transformation_plan", "run_script", "validate_csv",
                                             "compare_with_published"])
    for t in tools
        @test t.input_schema["type"] == "object"
        @test haskey(t.input_schema, "required")
        @test !isempty(t.description)
        @test t.annotations["readOnlyHint"] == (t.name != "run_script")
    end
    @test only(filter(t -> t.name == "run_script", tools)).annotations["destructiveHint"] == true
    @test only(filter(t -> t.name == "load_schema", tools)).annotations["openWorldHint"] == true

    prompts = SDMXerWizard.mcp_prompts()
    @test length(prompts) == 1
    @test prompts[1].name == "map_to_sdmx"
    @test "file" in [a.name for a in prompts[1].arguments if a.required]
    @test occursin("{file}", prompts[1].messages[1].content.text)
    @test occursin("list_dataflows", prompts[1].messages[1].content.text)
    load_tool = only(filter(t -> t.name == "load_source", tools))
    content = load_tool.handler(Dict{String, Any}("path" => DEMO_CSV))
    @test content isa TextContent
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
    proc = open(Base.pipeline(cmd; stderr=devnull), "r+")
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
    println(proc, JSON3.write(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "prompts/list"))); flush(proc)
    plist = JSON3.read(readline(proc), Dict{String, Any})
    @test [p["name"] for p in plist["result"]["prompts"]] == ["map_to_sdmx"]
    println(proc, JSON3.write(Dict("jsonrpc" => "2.0", "id" => 5, "method" => "prompts/get",
        "params" => Dict("name" => "map_to_sdmx", "arguments" => Dict("file" => "/tmp/x.csv", "keywords" => "inbound tourism"))))); flush(proc)
    pget = JSON3.read(readline(proc), Dict{String, Any})
    @test occursin("/tmp/x.csv", pget["result"]["messages"][1]["content"]["text"])
    @test !occursin("{endpoint}", pget["result"]["messages"][1]["content"]["text"])
    call = Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
        "params" => Dict("name" => "load_source", "arguments" => Dict("path" => DEMO_CSV)))
    println(proc, JSON3.write(call)); flush(proc)
    called = JSON3.read(readline(proc), Dict{String, Any})
    body = JSON3.read(called["result"]["content"][1]["text"], Dict{String, Any})
    @test body["source_id"] == "source_1"
    close(proc)
end
