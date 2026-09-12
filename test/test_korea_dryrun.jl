using Test
using SDMXer
using SDMXerWizard
using DataFrames
using CSV

# Regression tests for the issues found during the Korea workshop dry run
# (September 2026). Everything here runs offline: the target schema is a
# hand-built DataflowSchema with no codelists, so no codelist fetch happens.

@isdefined(fixture_schema) || include("fixtures.jl")

const DRYRUN_CSV = "Market,Visitor type,Year,Visitors ('000)\n" *
                   "China,Overnight visitors,2015,5984\n" *
                   "Japan,Overnight visitors,2015,1837\n" *
                   "China,Same-day visitors,2016,6117\n" *
                   "Japan,Same-day visitors,2016,2298\n"

@testset "Korea dry run regressions" begin
    schema = fixture_schema(with_codelists=false)

    @testset "infer_mappings method=:fuzzy" begin
        df = DataFrame(
            Market = ["China", "Japan", "China", "Japan"],
            var"Visitor type" = ["Overnight", "Overnight", "Same-day", "Same-day"],
            Year = [2015, 2015, 2016, 2016],
            var"Visitors ('000)" = [5984.0, 1837.0, 6117.0, 2298.0]
        )
        fuzzy = infer_mappings(df, schema; method=:fuzzy, confidence_threshold=0.2)
        @test fuzzy isa Dict{String, Vector{String}}
        @test haskey(fuzzy, "VISITOR_TYPE")
        @test "Visitor type" in fuzzy["VISITOR_TYPE"]

        # :advanced uses the same engine kwargs (no codelists in this schema, so offline)
        advanced = infer_mappings(df, schema; method=:advanced, confidence_threshold=0.2)
        @test advanced isa Dict{String, Vector{String}}
        @test haskey(advanced, "VISITOR_TYPE")

        # The engine flag behind use_codelists / :fuzzy
        @test create_inference_engine().use_value_matching == true
        @test create_inference_engine(use_value_matching=false).use_value_matching == false
    end

    @testset "validate_mapping_quality with and without schema" begin
        df = DataFrame(
            var"Visitor type" = ["Overnight", "Overnight", "Same-day", "Same-day"],
            Year = [2015, 2015, 2016, 2016]
        )
        profile = profile_source_data(df, "dryrun.csv")
        engine = create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        result = infer_advanced_mappings(engine, profile, schema, df)
        @test !isempty(result.unmapped_target_columns)

        v_noschema = validate_mapping_quality(result)
        @test haskey(v_noschema, "warnings")
        @test !haskey(v_noschema, "unmapped_required_columns")

        v_schema = validate_mapping_quality(result, schema)
        @test haskey(v_schema, "unmapped_required_columns")
        unmapped_required = v_schema["unmapped_required_columns"]
        @test unmapped_required == intersect(result.unmapped_target_columns, get_required_columns(schema))
        # OBS_STATUS is conditional, so it must never be reported as required
        @test !("OBS_STATUS" in unmapped_required)
        @test "REF_AREA" in unmapped_required
        @test any(w -> occursin("required columns remain unmapped", w), v_schema["warnings"])
    end

    @testset "CSV.jl column types are accepted" begin
        df = CSV.read(IOBuffer(DRYRUN_CSV), DataFrame; pool=true)
        # Sanity check that this exercises the non-Vector / non-String types
        @test !(df[!, :Market] isa Vector)
        @test !(eltype(df[!, :Market]) === String)

        @test fuzzy_match_score(df[1, :Market], "REF_AREA") isa Float64
        @test fuzzy_match_score(df[1, :Market], df[1, :Market]) == 1.0
        @test fuzzy_match_score(SubString("country_x", 1:7), "country") == 1.0

        codelist = DataFrame(code_id = ["CN", "JP"], name = ["China", "Japan"])
        patterns = analyze_value_patterns(df[!, :Market], codelist)
        @test patterns["fuzzy_matches"] + patterns["exact_matches"] == 2

        profile = profile_source_data(df, "dryrun.csv")
        engine = create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        result = infer_advanced_mappings(engine, profile, schema, df)
        @test result isa AdvancedMappingResult
        @test any(m -> m.source_column == "Visitor type" && m.target_column == "VISITOR_TYPE",
                  result.mappings)

        # Columns can be profiled without collect()
        col_profile = profile_column("Market", df[!, :Market])
        @test col_profile.name == "Market"
        @test col_profile.unique_count == 2
    end

    @testset "read_source_data parses the CSV header" begin
        path = tempname() * ".csv"
        write(path, DRYRUN_CSV)
        try
            df = read_source_data(path)
            @test names(df) == ["Market", "Visitor type", "Year", "Visitors ('000)"]
            @test nrow(df) == 4
            @test df[1, :Market] == "China"

            # skip_rows skips rows between the header and the data
            skipped = read_data(CSVSource(path; skip_rows=1))
            @test names(skipped) == ["Market", "Visitor type", "Year", "Visitors ('000)"]
            @test nrow(skipped) == 3
            @test skipped[1, :Market] == "Japan"
        finally
            rm(path; force=true)
        end
    end

    @testset "SDMXer entry points re-exported" begin
        for name in (:DataflowSchema, :extract_dataflow_schema, :extract_all_codelists,
                     :get_required_columns, :get_optional_columns)
            @test name in names(SDMXerWizard)
            @test getfield(SDMXerWizard, name) === getfield(SDMXer, name)
        end
    end
end
