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
