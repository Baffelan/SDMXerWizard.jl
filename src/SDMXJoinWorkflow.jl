"""
Cross-Dataflow Join Workflow for SDMXerWizard.jl

End-to-end workflow for fetching, comparing, harmonizing, and joining
multiple SDMX dataflows. Combines deterministic SDMXer operations with
optional LLM-assisted inference.
"""

# Dependencies loaded at package level

# =================== TYPES ===================

"""
    JoinWorkflowConfig

Configuration for an end-to-end cross-dataflow join workflow.

# Fields
- `dataflow_ids::Vector{String}`: IDs of dataflows to join
- `base_url::String`: SDMX API base URL
- `agency::String`: SDMX agency ID
- `research_question::String`: Research question guiding the analysis
- `time_range::Union{Tuple{String, String}, Nothing}`: Optional (start, end) time filter
- `geo_filter::Union{Vector{String}, Nothing}`: Optional geographic codes to filter
- `exchange_rates::Union{ExchangeRateTable, Nothing}`: Exchange rates for currency conversion
- `llm_provider::Symbol`: LLM provider for assisted inference
- `llm_model::String`: LLM model name
- `use_llm::Bool`: Whether to use LLM for ambiguous cases
- `dataflow_filters::Dict{String, Dict{String, Any}}`: Per-dataflow dimension filters (dataflow_id => dimension filters)

# See also
- [`execute_join_workflow`](@ref): runs this configuration end-to-end
- [`suggest_analysis_dataflows`](@ref): LLM-assisted selection of dataflow IDs
"""
struct JoinWorkflowConfig
    dataflow_ids::Vector{String}
    base_url::String
    agency::String
    research_question::String
    time_range::Union{Tuple{String, String}, Nothing}
    geo_filter::Union{Vector{String}, Nothing}
    exchange_rates::Union{ExchangeRateTable, Nothing}
    llm_provider::Symbol
    llm_model::String
    use_llm::Bool
    dataflow_filters::Dict{String, Dict{String, Any}}
end

function JoinWorkflowConfig(
        dataflow_ids::Vector{String};
        base_url::String = "https://stats-sdmx-disseminate.pacificdata.org/rest/",
        agency::String = "SPC",
        research_question::String = "",
        time_range::Union{Tuple{String, String}, Nothing} = nothing,
        geo_filter::Union{Vector{String}, Nothing} = nothing,
        exchange_rates::Union{ExchangeRateTable, Nothing} = nothing,
        llm_provider::Symbol = :ollama,
        llm_model::String = "",
        use_llm::Bool = false,
        dataflow_filters::Dict{String, Dict{String, Any}} = Dict{String, Dict{String, Any}}()
)
    return JoinWorkflowConfig(
        dataflow_ids, base_url, agency, research_question,
        time_range, geo_filter, exchange_rates,
        llm_provider, llm_model, use_llm, dataflow_filters
    )
end

# =================== WORKFLOW EXECUTION ===================

"""
    execute_join_workflow(config::JoinWorkflowConfig) -> Dict{String, Any}

Execute an end-to-end cross-dataflow join workflow:

1. Fetch schemas for all dataflows
2. Compare all schema pairs
3. Detect unit conflicts
4. Fetch data from each dataflow
5. Harmonize units
6. Align frequencies
7. Join all DataFrames sequentially
8. Optionally generate a join script via LLM

Returns a Dict with keys:
- `"schemas"`: Vector of DataflowSchema
- `"comparisons"`: Vector of SchemaComparison
- `"data"`: Dict mapping dataflow_id to DataFrame
- `"unit_reports"`: Vector of UnitConflictReport
- `"join_result"`: Final JoinResult (or nothing if < 2 dataflows with data)
- `"script"`: Generated join script (if use_llm=true)
- `"warnings"`: Vector of warning strings
- `"status"`: "success" or "error"

# See also
[`JoinWorkflowConfig`](@ref), `JoinResult` (SDMXer), [`generate_join_script`](@ref)
"""
function execute_join_workflow(config::JoinWorkflowConfig)
    result = Dict{String, Any}(
        "schemas" => DataflowSchema[],
        "comparisons" => SchemaComparison[],
        "data" => Dict{String, DataFrame}(),
        "unit_reports" => UnitConflictReport[],
        "join_result" => nothing,
        "script" => nothing,
        "warnings" => String[],
        "status" => "success"
    )

    warnings = result["warnings"]::Vector{String}

    # Step 1: Fetch schemas
    schemas = DataflowSchema[]
    schema_map = Dict{String, DataflowSchema}()
    for df_id in config.dataflow_ids
        url = config.base_url * "dataflow/" * config.agency * "/" * df_id * "/latest?references=all"
        try
            schema = extract_dataflow_schema(url)
            push!(schemas, schema)
            schema_map[df_id] = schema
        catch e
            push!(warnings, "Failed to fetch schema for " * df_id * ": " * string(e))
        end
    end
    result["schemas"] = schemas

    # Step 2: Compare all pairs
    comparisons = SchemaComparison[]
    for i in 1:length(schemas)
        for j in (i + 1):length(schemas)
            push!(comparisons, compare_schemas(schemas[i], schemas[j]))
        end
    end
    result["comparisons"] = comparisons

    # Step 3: Fetch data
    data_frames = Dict{String, DataFrame}()
    for df_id in config.dataflow_ids
        try
            # Build filters: start from per-dataflow filters, fold in legacy geo_filter
            df_filters = Dict{String, Any}(get(config.dataflow_filters, df_id, Dict{String, Any}()))
            if !isnothing(config.geo_filter) && !haskey(df_filters, "GEO_PICT")
                df_filters["GEO_PICT"] = config.geo_filter
            end

            df = query_sdmx_data(config.base_url, config.agency, df_id;
                filters = df_filters,
                start_period = isnothing(config.time_range) ? nothing : config.time_range[1],
                end_period = isnothing(config.time_range) ? nothing : config.time_range[2])

            data_frames[df_id] = df
        catch e
            push!(warnings, "Failed to fetch data for " * df_id * ": " * string(e))
        end
    end
    result["data"] = data_frames

    # Step 4: Sequential join
    df_ids_with_data = [id for id in config.dataflow_ids if haskey(data_frames, id)]

    if length(df_ids_with_data) < 2
        push!(warnings, "Need at least 2 dataflows with data to join, got " * string(length(df_ids_with_data)))
        result["status"] = length(df_ids_with_data) == 0 ? "error" : "partial"
        return result
    end

    # Join sequentially: first pair, then each additional
    accumulated = data_frames[df_ids_with_data[1]]
    schema_acc = get(schema_map, df_ids_with_data[1], nothing)

    for i in 2:length(df_ids_with_data)
        next_id = df_ids_with_data[i]
        next_df = data_frames[next_id]
        next_schema = get(schema_map, next_id, nothing)

        # Detect unit conflicts
        unit_report = detect_unit_conflicts(accumulated, next_df;
            exchange_rates = config.exchange_rates)
        push!(result["unit_reports"]::Vector{UnitConflictReport}, unit_report)

        suffix_a = i == 2 ? "_" * df_ids_with_data[1] : ""
        suffix_b = "_" * next_id

        join_result = sdmx_join(accumulated, next_df;
            join_type = :inner,
            validate_units = true,
            harmonize = true,
            exchange_rates = config.exchange_rates,
            schema_a = schema_acc,
            schema_b = next_schema,
            suffix_a = suffix_a,
            suffix_b = suffix_b)

        append!(warnings, join_result.warnings)
        accumulated = join_result.data

        if i == length(df_ids_with_data)
            result["join_result"] = join_result
        end
    end

    # Step 5: Optional LLM script generation
    if config.use_llm && !isempty(config.research_question)
        try
            script = generate_join_script(
                config.dataflow_ids, config.research_question;
                base_url = config.base_url,
                agency = config.agency,
                provider = config.llm_provider,
                model = config.llm_model
            )
            result["script"] = script
        catch e
            push!(warnings, "LLM script generation failed: " * string(e))
        end
    end

    return result
end
