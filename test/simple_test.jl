using Test
using SDMXer
using SDMXerWizard
using PromptingTools
using DataFrames

@testset "SDMXerWizard Basic Tests" begin

    @testset "LLM Setup" begin
        # Test that setup_sdmx_llm works
        provider = setup_sdmx_llm(:ollama; model="llama2")
        @test provider == SDMXerWizard.OLLAMA

        # Test that enums are exported
        @test SDMXerWizard.OLLAMA isa SDMXerWizard.LLMProvider
        @test SDMXerWizard.OPENAI isa SDMXerWizard.LLMProvider

        # Test automatic Responses API schema selection for response-only models
        response_schema = SDMXerWizard._select_openai_schema("gpt-5.1-codex-mini")
        @test response_schema isa PromptingTools.OpenAIResponseSchema

        chat_schema = SDMXerWizard._select_openai_schema("gpt-4o")
        @test chat_schema isa PromptingTools.OpenAISchema
    end

    @testset "Data Sources" begin
        # Test CSV source with actual file
        test_file = "/home/gvdr/reps/julia_sdmx/SDMXer.jl/test/fixtures/sample_data.csv"
        if isfile(test_file)
            csv_source = CSVSource(test_file)
            @test csv_source.path == test_file
            @test csv_source isa DataSource
        else
            @test_skip "Test file not found"
        end

        # Test DataFrame source
        df = DataFrame(a=[1,2,3], b=[4,5,6])
        df_source = DataFrameSource(df, "test_df")
        @test df_source.data == df
        @test df_source.name == "test_df"
    end

    @testset "Basic Functions" begin
        # Test that key functions are exported and accessible
        @test isdefined(SDMXerWizard, :setup_sdmx_llm)
        @test isdefined(SDMXerWizard, :create_workflow)
        @test isdefined(SDMXerWizard, :generate_transformation_script)
        @test isdefined(SDMXerWizard, :infer_column_mappings)
        @test isdefined(SDMXerWizard, :create_inference_engine)
        @test isdefined(SDMXerWizard, :anonymize_source_data)
        @test isdefined(SDMXerWizard, :summarize_anonymized_data)
        @test isdefined(SDMXerWizard, :AnonymizationConfig)
    end

    @testset "Inference Engine" begin
        # Test inference engine creation
        engine = create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        @test engine.fuzzy_threshold == 0.6
        @test engine.min_confidence == 0.2
    end

    @testset "Codelist Loading" begin
        url = "https://stats-sdmx-disseminate.pacificdata.org/rest/datastructure/SPC/DF_BP50"
        schema = extract_dataflow_schema(url)
        engine = create_inference_engine()
        SDMXerWizard.load_codelist_data!(engine, schema)
        @test length(engine.codelists_data) > 0
    end

    @testset "Anonymization" begin
        df = DataFrame(country=["FJ", "TV"], year=[2020, 2021], value=[1.0, 2.0])
        profile = profile_source_data(df, "test.csv")
        anon = anonymize_source_data(df, profile)
        @test nrow(anon) == nrow(df)
        @test names(anon) == names(df)

        summary = summarize_anonymized_data(anon; max_samples=3)
        @test length(summary.columns) == ncol(df)

        @testset ":preserve_distribution mode" begin
            df2 = DataFrame(
                category=["A", "B", "A", "C", "B"],
                value=[10.0, 20.0, 30.0, 40.0, 50.0]
            )
            profile2 = profile_source_data(df2, "test.csv")
            cfg = AnonymizationConfig(:preserve_distribution, 1000, 50, true, true)
            anon2 = anonymize_source_data(df2, profile2; config=cfg)

            @test nrow(anon2) == nrow(df2)
            @test names(anon2) == names(df2)

            value_tokens = anon2[!, :value]
            @test all(v -> startswith(string(v), "NUM_Q"), value_tokens)
            @test length(unique(value_tokens)) <= 10

            category_tokens = anon2[!, :category]
            @test all(v -> startswith(string(v), "CAT_"), category_tokens)
        end
    end

    @testset "Fuzzy Matching" begin
        # Test fuzzy match scoring
        score = fuzzy_match_score("country", "COUNTRY")
        @test score > 0.9  # Should be very high for case differences

        score2 = fuzzy_match_score("geo", "GEO_PICT")
        @test score2 > 0.3  # Should find some similarity
    end

end

println("\n✓ All basic tests passed!")
