using Test, JSON3, DataFrames, CSV
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
    @test summary["codelists_loaded"] == true
    @test roundtrip(summary)["schema_id"] == id

    all_codes = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "VISITOR_TYPE"))
    @test all_codes["total_codes"] == 2
    @test [m["code"] for m in all_codes["matches"]] == ["OVN", "SDV"]
    @test all_codes["truncated"] == false

    hit = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "REF_AREA", "query" => "japan"))
    @test hit["matches"][1]["code"] == "JP"

    attr = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "UNIT_MEASURE"))
    @test attr["codelist_id"] == "CL_UNIT"

    none = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "COUNTERPART_AREA"))
    @test haskey(none, "error")

    limited = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => id, "dimension" => "REF_AREA", "limit" => 1))
    @test length(limited["matches"]) == 1
    @test limited["truncated"] == true

    nocl = Tools.register_schema!(session, fixture_schema(), nothing; origin="fixture-nocodes")
    without = Tools.run_tool(session, Tools.lookup_codes, Dict("schema_id" => nocl, "dimension" => "REF_AREA"))
    @test occursin("with_codelists", without["hint"])
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
