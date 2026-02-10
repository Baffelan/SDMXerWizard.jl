"""
Prompt Construction Helpers for Cross-Dataflow Analysis

Builds structured prompts for LLM-assisted cross-dataflow operations.
No LLM calls — just string building. Used by SDMXCrossDataflowLLM.jl.
"""

# Dependencies loaded at package level

# =================== PROMPT BUILDERS ===================

"""
    create_join_analysis_prompt(schemas::Vector{DataflowSchema},
                               comparisons::Vector{SchemaComparison},
                               research_question::String) -> String

Build a prompt that describes multiple dataflow schemas and their comparisons,
asking the LLM to recommend a join strategy for a given research question.

# See also
[`sdmx_aigenerate`](@ref), `compare_schemas` (SDMXer)
"""
function create_join_analysis_prompt(
        schemas::Vector{DataflowSchema},
        comparisons::Vector{SchemaComparison},
        research_question::String
)
    parts = String[]

    push!(parts, "# Cross-Dataflow Join Analysis\n")
    push!(parts, "## Research Question\n" * research_question * "\n")

    # Describe each schema
    push!(parts, "## Available Dataflows\n")
    for (i, schema) in enumerate(schemas)
        info = schema.dataflow_info
        name = ismissing(info.name) ? info.id : info.name
        push!(parts, "### Dataflow " * string(i) * ": " * string(name) * "\n")
        push!(parts, "- ID: " * string(info.id) * "\n")
        push!(parts, "- Agency: " * string(info.agency) * "\n")

        dim_ids = join(schema.dimensions.dimension_id, ", ")
        push!(parts, "- Dimensions: " * dim_ids * "\n")

        if !isempty(schema.attributes)
            attr_ids = join(schema.attributes.attribute_id, ", ")
            push!(parts, "- Attributes: " * attr_ids * "\n")
        end

        if !isnothing(schema.time_dimension)
            push!(parts, "- Time dimension: " * schema.time_dimension.dimension_id * "\n")
        end
        push!(parts, "")
    end

    # Describe comparisons
    if !isempty(comparisons)
        push!(parts, "## Schema Comparisons\n")
        for comp in comparisons
            a_id = string(comp.schema_a_info.id)
            b_id = string(comp.schema_b_info.id)
            push!(parts, "### " * a_id * " vs " * b_id * "\n")
            push!(parts, "- Joinability score: " * string(round(comp.joinability_score; digits = 2)) * "\n")
            push!(parts, "- Recommended join dims: " * join(comp.recommended_join_dims, ", ") * "\n")
            push!(parts, "- Unique to " * a_id * ": " * join(comp.unique_to_a, ", ") * "\n")
            push!(parts, "- Unique to " * b_id * ": " * join(comp.unique_to_b, ", ") * "\n")
            push!(parts, "")
        end
    end

    push!(parts, "## Task\n")
    push!(parts, "Based on the schema analysis above, recommend:\n")
    push!(parts, "1. Which dataflows to join and in what order\n")
    push!(parts, "2. Which dimensions to join on\n")
    push!(parts, "3. What join type to use (inner, left, outer)\n")
    push!(parts, "4. Any unit or frequency alignment needed\n")
    push!(parts, "5. Potential data quality issues to watch for\n")

    return join(parts, "")
end

"""
    create_indicator_classification_prompt(indicators_df::DataFrame,
                                          dataflow_id::String) -> String

Build a prompt asking the LLM to classify SDMX indicators by semantic type
(volume, value, price, share, rate, index).

# See also
[`sdmx_aigenerate`](@ref), [`infer_indicator_semantics`](@ref)
"""
function create_indicator_classification_prompt(
        indicators_df::DataFrame, dataflow_id::String)
    parts = String[]

    push!(parts, "# Indicator Classification Task\n")
    push!(parts, "Dataflow: " * dataflow_id * "\n\n")
    push!(parts, "Classify each indicator code below into one of these categories:\n")
    push!(parts, "- volume: Physical quantity or count\n")
    push!(parts, "- value: Monetary amount\n")
    push!(parts, "- price: Price or unit value\n")
    push!(parts, "- share: Percentage or proportion\n")
    push!(parts, "- rate: Rate of change or ratio\n")
    push!(parts, "- index: Index number\n")
    push!(parts, "- other: Does not fit above categories\n\n")

    push!(parts, "## Indicators\n")
    for row in eachrow(indicators_df)
        code = hasproperty(indicators_df, :code_id) ? string(row.code_id) : string(row[1])
        name = if hasproperty(indicators_df, :name)
            string(row.name)
        elseif ncol(indicators_df) >= 2
            string(row[2])
        else
            ""
        end
        push!(parts, "- " * code * ": " * name * "\n")
    end

    push!(parts, "\nReturn a JSON object mapping each indicator code to its category.\n")

    return join(parts, "")
end

"""
    create_unit_inference_prompt(unit_a::String, unit_b::String,
                                context::Dict{String, Any}) -> String

Build a prompt asking the LLM to interpret ambiguous SDMX unit metadata.
Context should include value ranges, dataflow names, and any other hints.

# See also
[`sdmx_aigenerate`](@ref), [`infer_unit_conversion`](@ref)
"""
function create_unit_inference_prompt(
        unit_a::String, unit_b::String, context::Dict{String, Any})
    parts = String[]

    push!(parts, "# Unit Inference Task\n\n")
    push!(parts, "Two SDMX dataflows use different unit codes. Help determine if they are comparable.\n\n")
    push!(parts, "## Unit A: " * unit_a * "\n")
    push!(parts, "## Unit B: " * unit_b * "\n\n")

    if haskey(context, "dataflow_a")
        push!(parts, "Dataflow A: " * string(context["dataflow_a"]) * "\n")
    end
    if haskey(context, "dataflow_b")
        push!(parts, "Dataflow B: " * string(context["dataflow_b"]) * "\n")
    end
    if haskey(context, "value_range_a")
        push!(parts, "Value range A: " * string(context["value_range_a"]) * "\n")
    end
    if haskey(context, "value_range_b")
        push!(parts, "Value range B: " * string(context["value_range_b"]) * "\n")
    end
    if haskey(context, "description")
        push!(parts, "\nAdditional context: " * string(context["description"]) * "\n")
    end

    push!(parts, "\n## Task\n")
    push!(parts, "1. Are these units measuring the same physical quantity?\n")
    push!(parts, "2. If so, what is the likely conversion factor?\n")
    push!(parts, "3. Could the value ranges indicate a hidden unit multiplier (e.g., thousands vs units)?\n")
    push!(parts, "4. What confidence level do you have in this assessment?\n")
    push!(parts, "\nReturn a JSON with: comparable (bool), conversion_factor (float or null), ")
    push!(parts, "confidence (0-1), reasoning (string).\n")

    return join(parts, "")
end
