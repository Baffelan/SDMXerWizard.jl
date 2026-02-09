"""
LLM-Assisted Cross-Dataflow Analysis for SDMXerWizard.jl

Adds LLM inference for ambiguous cross-dataflow scenarios:
- Unit interpretation when metadata is ambiguous
- Dataflow suggestions for research questions
- Join script generation
- Indicator semantic classification
- Cross-dataflow indicator comparison
"""

# Dependencies loaded at package level

# =================== LLM-ASSISTED FUNCTIONS ===================

"""
    infer_unit_conversion(unit_a::String, unit_b::String;
                         context::Dict{String, Any}=Dict{String, Any}(),
                         value_range_a::Union{Tuple{Float64, Float64}, Nothing}=nothing,
                         value_range_b::Union{Tuple{Float64, Float64}, Nothing}=nothing,
                         provider::Symbol=:ollama,
                         model::String="") -> Dict{String, Any}

Use an LLM to interpret ambiguous SDMX unit metadata. Useful when both dataflows
use similar codes but value ranges suggest different actual units (e.g., both tagged
"FJD" but one is FJD/kg and other FJD/tonne).

Returns a Dict with keys: "comparable", "conversion_factor", "confidence", "reasoning".
"""
function infer_unit_conversion(
        unit_a::String, unit_b::String;
        context::Dict{String, Any} = Dict{String, Any}(),
        value_range_a::Union{Tuple{Float64, Float64}, Nothing} = nothing,
        value_range_b::Union{Tuple{Float64, Float64}, Nothing} = nothing,
        provider::Symbol = :ollama,
        model::String = ""
)
    # Build context
    if !isnothing(value_range_a)
        context["value_range_a"] = string(value_range_a[1]) * " to " * string(value_range_a[2])
    end
    if !isnothing(value_range_b)
        context["value_range_b"] = string(value_range_b[1]) * " to " * string(value_range_b[2])
    end

    prompt = create_unit_inference_prompt(unit_a, unit_b, context)

    config = setup_sdmx_llm(provider; model = model)
    result = sdmx_aigenerate(config, prompt)

    # Parse response — try to extract structured data
    response_text = if result isa AbstractString
        result
    elseif hasproperty(result, :content)
        result.content
    else
        string(result)
    end

    return Dict{String, Any}(
        "comparable" => occursin(r"true|yes|comparable"i, response_text),
        "conversion_factor" => nothing,
        "confidence" => 0.5,
        "reasoning" => response_text,
        "raw_response" => response_text
    )
end

"""
    suggest_analysis_dataflows(research_question::String;
                              base_url::String="https://stats-sdmx-disseminate.pacificdata.org/rest/",
                              agency::String="SPC",
                              provider::Symbol=:ollama,
                              model::String="") -> Vector{Dict{String, Any}}

Ask an LLM to suggest relevant SDMX dataflows for a research question.

Returns a vector of Dicts with keys: "dataflow_id", "relevance", "reasoning".
"""
function suggest_analysis_dataflows(
        research_question::String;
        base_url::String = "https://stats-sdmx-disseminate.pacificdata.org/rest/",
        agency::String = "SPC",
        provider::Symbol = :ollama,
        model::String = ""
)
    prompt = "# SDMX Dataflow Suggestion\n\n" *
             "Research question: " * research_question * "\n\n" *
             "Agency: " * agency * "\n" *
             "Base URL: " * base_url * "\n\n" *
             "Suggest 3-5 SDMX dataflows from " * agency * " that would be relevant " *
             "for this research question. For each, explain why it is relevant and " *
             "how it connects to the others.\n\n" *
             "Return a JSON array of objects with: dataflow_id, relevance (high/medium/low), reasoning."

    config = setup_sdmx_llm(provider; model = model)
    result = sdmx_aigenerate(config, prompt)

    response_text = if result isa AbstractString
        result
    elseif hasproperty(result, :content)
        result.content
    else
        string(result)
    end

    return [Dict{String, Any}(
        "response" => response_text,
        "research_question" => research_question,
        "agency" => agency
    )]
end

"""
    generate_join_script(dataflow_ids::Vector{String},
                        research_question::String;
                        base_url::String="https://stats-sdmx-disseminate.pacificdata.org/rest/",
                        agency::String="SPC",
                        provider::Symbol=:ollama,
                        model::String="") -> GeneratedScript

Generate a Julia script that fetches, normalizes, and joins multiple SDMX dataflows.
"""
function generate_join_script(
        dataflow_ids::Vector{String},
        research_question::String;
        base_url::String = "https://stats-sdmx-disseminate.pacificdata.org/rest/",
        agency::String = "SPC",
        provider::Symbol = :ollama,
        model::String = ""
)
    # Fetch schemas for context
    schemas = DataflowSchema[]
    comparisons = SchemaComparison[]
    for df_id in dataflow_ids
        url = base_url * "dataflow/" * agency * "/" * df_id * "/latest?references=all"
        try
            schema = extract_dataflow_schema(url)
            push!(schemas, schema)
        catch e
            @warn "Could not fetch schema for " * df_id * ": " * string(e)
        end
    end

    # Compare all pairs
    for i in 1:length(schemas)
        for j in (i + 1):length(schemas)
            push!(comparisons, compare_schemas(schemas[i], schemas[j]))
        end
    end

    join_prompt = create_join_analysis_prompt(schemas, comparisons, research_question)

    script_prompt = join_prompt * "\n\n" *
                    "## Script Generation Task\n\n" *
                    "Generate a complete Julia script using SDMXer.jl that:\n" *
                    "1. Fetches data from each dataflow using query_sdmx_data()\n" *
                    "2. Normalizes units using normalize_units!()\n" *
                    "3. Aligns frequencies if needed using align_frequencies()\n" *
                    "4. Joins the dataflows using sdmx_join()\n" *
                    "5. Saves the result to CSV\n\n" *
                    "Use string concatenation (*) instead of interpolation. " *
                    "Include error handling and comments."

    config = setup_sdmx_llm(provider; model = model)
    result = sdmx_aigenerate(config, script_prompt)

    response_text = if result isa AbstractString
        result
    elseif hasproperty(result, :content)
        result.content
    else
        string(result)
    end

    dataflow_label = join(dataflow_ids, "+")

    return GeneratedScript(
        "cross_dataflow_join",
        "",
        dataflow_label,
        response_text,
        TransformationStep[],
        0.5,
        String[],
        String[],
        "cross_dataflow",
        string(Dates.now())
    )
end

"""
    infer_indicator_semantics(dataflow_id::String,
                             indicator_codes::Vector{String};
                             base_url::String="https://stats-sdmx-disseminate.pacificdata.org/rest/",
                             agency::String="SPC",
                             provider::Symbol=:ollama,
                             model::String="") -> Dict{String, String}

Use an LLM to classify indicators as volume/value/price/share/rate/index.
Returns a Dict mapping indicator code to category string.
"""
function infer_indicator_semantics(
        dataflow_id::String,
        indicator_codes::Vector{String};
        base_url::String = "https://stats-sdmx-disseminate.pacificdata.org/rest/",
        agency::String = "SPC",
        provider::Symbol = :ollama,
        model::String = ""
)
    # Build an indicator DataFrame for the prompt
    indicators_df = DataFrame(code_id = indicator_codes, name = indicator_codes)

    # Try to fetch actual names from codelists
    try
        url = base_url * "dataflow/" * agency * "/" * dataflow_id * "/latest?references=all"
        schema = extract_dataflow_schema(url)
        codelists = extract_all_codelists(url)
        # Find indicator codelist
        indicator_dims = filter(r -> occursin("INDICATOR", uppercase(string(r.dimension_id))), eachrow(schema.dimensions))
        if !isempty(indicator_dims)
            cl_id = first(indicator_dims).codelist_id
            if !ismissing(cl_id)
                cl_codes = filter(r -> r.codelist_id == cl_id && r.lang == "en", codelists)
                name_map = Dict(string(r.code_id) => string(r.name) for r in eachrow(cl_codes))
                indicators_df = DataFrame(
                    code_id = indicator_codes,
                    name = [get(name_map, c, c) for c in indicator_codes]
                )
            end
        end
    catch e
        @warn "Could not fetch indicator names: " * string(e)
    end

    prompt = create_indicator_classification_prompt(indicators_df, dataflow_id)

    config = setup_sdmx_llm(provider; model = model)
    result = sdmx_aigenerate(config, prompt)

    response_text = if result isa AbstractString
        result
    elseif hasproperty(result, :content)
        result.content
    else
        string(result)
    end

    # Return raw response — caller can parse JSON
    classifications = Dict{String, String}()
    for code in indicator_codes
        classifications[code] = "unknown"
    end
    classifications["_raw_response"] = response_text

    return classifications
end

"""
    suggest_comparable_indicators(indicator::String,
                                 dataflow_ids::Vector{String};
                                 base_url::String="https://stats-sdmx-disseminate.pacificdata.org/rest/",
                                 agency::String="SPC",
                                 provider::Symbol=:ollama,
                                 model::String="") -> Vector{Dict{String, Any}}

Find semantically similar indicators across multiple dataflows using LLM analysis.
"""
function suggest_comparable_indicators(
        indicator::String,
        dataflow_ids::Vector{String};
        base_url::String = "https://stats-sdmx-disseminate.pacificdata.org/rest/",
        agency::String = "SPC",
        provider::Symbol = :ollama,
        model::String = ""
)
    prompt = "# Comparable Indicator Search\n\n" *
             "Find indicators similar to '" * indicator * "' across these SDMX dataflows:\n"

    for df_id in dataflow_ids
        prompt = prompt * "- " * df_id * "\n"
    end

    prompt = prompt * "\nFor each match, provide: dataflow_id, indicator_code, " *
             "similarity_score (0-1), reasoning.\n" *
             "Return as JSON array."

    config = setup_sdmx_llm(provider; model = model)
    result = sdmx_aigenerate(config, prompt)

    response_text = if result isa AbstractString
        result
    elseif hasproperty(result, :content)
        result.content
    else
        string(result)
    end

    return [Dict{String, Any}(
        "response" => response_text,
        "query_indicator" => indicator,
        "dataflow_ids" => dataflow_ids
    )]
end
