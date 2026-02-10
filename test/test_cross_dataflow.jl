using Test
using SDMXer
using SDMXerWizard
using DataFrames

@testset "Cross-Dataflow LLM" begin
    @testset "Prompt Construction — join analysis" begin
        # Create mock schemas
        bp50_schema = extract_dataflow_schema(
            "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all"
        )

        comparison = compare_schemas(bp50_schema, bp50_schema)

        prompt = create_join_analysis_prompt(
            [bp50_schema],
            [comparison],
            "How does kava trade relate to national accounts?"
        )

        @test prompt isa String
        @test length(prompt) > 100
        @test occursin("Research Question", prompt)
        @test occursin("kava", prompt)
        @test occursin("Dataflow", prompt)
    end

    @testset "Prompt Construction — indicator classification" begin
        indicators = DataFrame(
            code_id = ["GDP", "CPI", "POP"],
            name = ["Gross Domestic Product", "Consumer Price Index", "Population"]
        )

        prompt = create_indicator_classification_prompt(indicators, "DF_NATACCOUNT")

        @test prompt isa String
        @test occursin("GDP", prompt)
        @test occursin("CPI", prompt)
        @test occursin("volume", prompt)
        @test occursin("value", prompt)
        @test occursin("index", prompt)
    end

    @testset "Prompt Construction — unit inference" begin
        context = Dict{String, Any}(
            "dataflow_a" => "DF_TRADE",
            "dataflow_b" => "DF_GDP",
            "value_range_a" => "0.1 to 50.0",
            "value_range_b" => "100 to 50000"
        )

        prompt = create_unit_inference_prompt("FJD", "FJD", context)

        @test prompt isa String
        @test occursin("FJD", prompt)
        @test occursin("DF_TRADE", prompt)
        @test occursin("conversion", prompt)
    end

    @testset "JoinWorkflowConfig construction" begin
        config = JoinWorkflowConfig(
            ["DF_BP50", "DF_POP"];
            agency = "SPC",
            research_question = "Trade vs population",
            use_llm = false
        )

        @test config.dataflow_ids == ["DF_BP50", "DF_POP"]
        @test config.agency == "SPC"
        @test config.research_question == "Trade vs population"
        @test config.use_llm == false
        @test isnothing(config.time_range)
        @test isnothing(config.geo_filter)
        @test isempty(config.dataflow_filters)
    end

    @testset "JoinWorkflowConfig with dataflow_filters" begin
        filters = Dict("DF_BP50" => Dict{String, Any}("INDICATOR" => "BP50_01"))
        config = JoinWorkflowConfig(
            ["DF_BP50"];
            dataflow_filters = filters
        )
        @test haskey(config.dataflow_filters, "DF_BP50")
        @test config.dataflow_filters["DF_BP50"]["INDICATOR"] == "BP50_01"
    end

    # LLM-dependent tests — guarded by API key availability
    @testset "LLM Functions (API key guarded)" begin
        has_openai = haskey(ENV, "OPENAI_API_KEY")
        has_anthropic = haskey(ENV, "ANTHROPIC_API_KEY")
        has_any_key = has_openai || has_anthropic

        if has_any_key
            provider = has_openai ? :openai : :anthropic

            @testset "infer_unit_conversion" begin
                result = infer_unit_conversion("FJD", "USD";
                    context = Dict{String, Any}("description" => "Trade data"),
                    provider = provider)
                @test result isa Dict
                @test haskey(result, "comparable")
                @test haskey(result, "reasoning")
            end
        else
            @info "Skipping LLM-dependent tests (no API keys in environment)"
        end
    end
end
