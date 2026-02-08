"""
Enhanced SDMX Transformation Script Generation

This module generates comprehensive transformation scripts using rich metadata context
without sending actual data to LLMs. It handles code mapping, pivoting, range selection,
and all complex transformation scenarios.
"""

# Dependencies loaded at package level


# =================== TRANSFORMATION CONTEXT BUILDER ===================

"""
    build_transformation_context(sdmx_context::SDMXStructuralContext,
                                source_context::DataSourceContext) -> TransformationContext

Combine SDMX and source contexts with automated mapping suggestions and transformation requirements.
"""
function build_transformation_context(sdmx_context::SDMXStructuralContext,
                                     source_context::DataSourceContext)
    @assert !isempty(sdmx_context.required_columns) "SDMX context must have required columns"
    @assert !isempty(source_context.source_profile.columns) "Source context must have column information"

    # Generate automated mapping suggestions using structural analysis
    mapping_suggestions = generate_mapping_suggestions(sdmx_context, source_context)
    @assert nrow(mapping_suggestions) > 0 "Must generate at least one mapping suggestion"

    # Identify required transformations
    transformation_requirements = identify_transformation_requirements(sdmx_context, source_context, mapping_suggestions)
    @assert !isempty(transformation_requirements) "Must identify transformation requirements"

    # Assess overall complexity
    complexity_assessment = assess_transformation_complexity(sdmx_context, source_context, transformation_requirements)
    @assert haskey(complexity_assessment, :overall_score) "Complexity assessment must include overall score"

    return TransformationContext(
        sdmx_context,
        source_context,
        mapping_suggestions,
        transformation_requirements,
        complexity_assessment
    )
end

# =================== AUTOMATED MAPPING SUGGESTIONS ===================

"""
Generate intelligent mapping suggestions based on structural analysis
"""
function generate_mapping_suggestions(sdmx_context::SDMXStructuralContext,
                                     source_context::DataSourceContext)
    suggestions = DataFrame(
        source_column = String[],
        target_column = String[],
        mapping_type = String[],
        confidence = Float64[],
        transformation_needed = String[],
        codelist_info = Union{String, Nothing}[],
        notes = String[]
    )

    source_columns = [col.name for col in source_context.source_profile.columns]
    @assert !isempty(source_columns) "Source must have columns"

    # Map temporal columns
    for time_pattern in source_context.time_patterns
        temporal_mapping = create_temporal_mapping(time_pattern, sdmx_context)
        if temporal_mapping !== nothing
            push!(suggestions, temporal_mapping)
        end
    end

    # Map geographic columns
    for geo_col in source_context.geographic_patterns
        geographic_mapping = create_geographic_mapping(geo_col, sdmx_context)
        if geographic_mapping !== nothing
            push!(suggestions, geographic_mapping)
        end
    end

    # Map value columns
    value_mappings = create_value_mappings(source_context, sdmx_context)
    for mapping in value_mappings
        push!(suggestions, mapping)
    end

    # Map categorical columns to dimensions
    categorical_mappings = create_categorical_mappings(source_context, sdmx_context)
    for mapping in categorical_mappings
        push!(suggestions, mapping)
    end

    # Add unmapped required columns as manual mapping needed
    mapped_targets = Set(suggestions.target_column)
    for required_col in sdmx_context.required_columns
        if required_col ∉ mapped_targets
            push!(suggestions, (
                source_column = "MANUAL_MAPPING_NEEDED",
                target_column = required_col,
                mapping_type = "manual",
                confidence = 0.0,
                transformation_needed = "user_specification",
                codelist_info = get(sdmx_context.codelist_columns, required_col, nothing),
                notes = "Requires manual specification of source column"
            ))
        end
    end

    return suggestions
end

function create_temporal_mapping(time_pattern, sdmx_context)
    # Find time dimension in SDMX schema
    time_dim = sdmx_context.time_dimension
    if time_dim === nothing
        return nothing
    end

    @assert !isempty(time_pattern.column) "Time pattern must have column name"

    transformation = if time_pattern.format == "unknown"
        "parse_and_standardize_temporal_format"
    else
        "convert_to_sdmx_time_format"
    end

    return (
        source_column = time_pattern.column,
        target_column = time_dim.dimension_id,
        mapping_type = "temporal",
        confidence = 0.9,
        transformation_needed = transformation,
        codelist_info = nothing,
        notes = "Detected temporal pattern: $(time_pattern.format)"
    )
end

function create_geographic_mapping(geo_col, sdmx_context)
    # Find geographic dimension
    geo_dims = filter(row -> occursin("GEO", uppercase(row.dimension_id)) ||
                           occursin("AREA", uppercase(row.dimension_id)) ||
                           occursin("COUNTRY", uppercase(row.dimension_id)),
                     sdmx_context.dimensions)

    if nrow(geo_dims) == 0
        return nothing
    end

    target_dim = geo_dims[1, :dimension_id]
    codelist_id = geo_dims[1, :codelist_id]

    @assert !isempty(geo_col) "Geographic column name cannot be empty"

    return (
        source_column = geo_col,
        target_column = target_dim,
        mapping_type = "geographic",
        confidence = 0.8,
        transformation_needed = "map_to_sdmx_geographic_codes",
        codelist_info = codelist_id,
        notes = "Geographic dimension mapping with code conversion needed"
    )
end

function create_value_mappings(source_context, sdmx_context)
    mappings = []

    # Find numeric columns in source
    numeric_cols = [col.name for col in source_context.source_profile.columns
                   if col.numeric_stats !== nothing]

    # Map to measures
    for (i, measure_row) in enumerate(eachrow(sdmx_context.measures))
        if i <= length(numeric_cols)
            source_col = numeric_cols[i]
            value_pattern = get(source_context.value_patterns, source_col, nothing)

            transformation = determine_value_transformation(value_pattern, measure_row)

            push!(mappings, (
                source_column = source_col,
                target_column = measure_row.measure_id,
                mapping_type = "measure",
                confidence = 0.7,
                transformation_needed = transformation,
                codelist_info = nothing,
                notes = "Numeric value mapping: $(value_pattern !== nothing ? value_pattern.scale : "unknown scale")"
            ))
        end
    end

    return mappings
end

function create_categorical_mappings(source_context, sdmx_context)
    mappings = []

    # Find categorical columns
    categorical_cols = [col for col in source_context.source_profile.columns
                       if col.is_categorical && col.name ∉ source_context.geographic_patterns]

    # Map to remaining dimensions
    available_dims = copy(sdmx_context.dimensions)

    for (i, col) in enumerate(categorical_cols)
        if i <= nrow(available_dims)
            dim_row = available_dims[i, :]
            codelist_id = dim_row.codelist_id

            confidence = calculate_categorical_confidence(col, codelist_id, sdmx_context)

            push!(mappings, (
                source_column = col.name,
                target_column = dim_row.dimension_id,
                mapping_type = "categorical",
                confidence = confidence,
                transformation_needed = "map_categorical_codes",
                codelist_info = codelist_id,
                notes = "Categorical mapping: $(col.unique_count) unique values"
            ))
        end
    end

    return mappings
end

# =================== TRANSFORMATION REQUIREMENTS IDENTIFICATION ===================

"""
Identify all required transformations based on context analysis
"""
function identify_transformation_requirements(sdmx_context, source_context, mappings)
    requirements = []

    # Check if pivoting is needed
    if source_context.data_shape.needs_pivoting
        pivoting_req = create_pivoting_requirement(source_context, mappings)
        push!(requirements, pivoting_req)
    end

    # Check if range selection is needed (Excel files)
    if source_context.excel_structure !== nothing
        range_req = create_range_selection_requirement(source_context.excel_structure)
        push!(requirements, range_req)
    end

    # Check if code mapping is needed
    code_mapping_reqs = create_code_mapping_requirements(sdmx_context, mappings)
    append!(requirements, code_mapping_reqs)

    # Check if data validation is needed
    validation_req = create_validation_requirement(sdmx_context, source_context)
    push!(requirements, validation_req)

    # Check if header cleanup is needed
    if !isempty(source_context.data_shape.header_issues)
        header_req = create_header_cleanup_requirement(source_context.data_shape.header_issues)
        push!(requirements, header_req)
    end

    return requirements
end

function create_pivoting_requirement(source_context, mappings)
    @assert source_context.data_shape.needs_pivoting "Should only be called when pivoting is needed"

    pivot_type = source_context.data_shape.is_wide_format ? "wide_to_long" : "long_to_wide"

    return (
        type = "pivoting",
        subtype = pivot_type,
        priority = "high",
        columns_involved = source_context.data_shape.pivot_candidates,
        instructions = "Reshape data from $pivot_type format",
        validation_needed = true
    )
end

function create_range_selection_requirement(excel_structure)
    @assert excel_structure !== nothing "Excel structure cannot be nothing"

    return (
        type = "range_selection",
        subtype = "excel_range",
        priority = "high",
        columns_involved = String[],
        instructions = "Select data range from Excel sheet: $(excel_structure.recommended_sheet)",
        validation_needed = true
    )
end

function create_code_mapping_requirements(sdmx_context, mappings)
    requirements = []

    for row in eachrow(mappings)
        if row.mapping_type in ["geographic", "categorical"] && !isnothing(row.codelist_info)
            codelist_id = row.codelist_info
            @assert haskey(sdmx_context.codelist_summary, codelist_id) "Codelist must exist in summary"

            codelist_info = sdmx_context.codelist_summary[codelist_id]

            req = (
                type = "code_mapping",
                subtype = row.mapping_type,
                priority = "medium",
                columns_involved = [row.source_column],
                instructions = "Map $(row.source_column) codes to $(codelist_id) codelist ($(codelist_info.total_codes) codes)",
                validation_needed = true,
                codelist_details = codelist_info
            )
            push!(requirements, req)
        end
    end

    return requirements
end

function create_validation_requirement(sdmx_context, source_context)
    return (
        type = "validation",
        subtype = "sdmx_compliance",
        priority = "high",
        columns_involved = sdmx_context.required_columns,
        instructions = "Validate output against SDMX schema requirements",
        validation_needed = false  # This IS the validation
    )
end

function create_header_cleanup_requirement(header_issues)
    @assert !isempty(header_issues) "Should only be called when header issues exist"

    return (
        type = "header_cleanup",
        subtype = "remove_metadata_rows",
        priority = "high",
        columns_involved = String[],
        instructions = "Remove $(length(header_issues)) metadata/header rows: $(join(header_issues, ", "))",
        validation_needed = false
    )
end

# =================== COMPLEXITY ASSESSMENT ===================

"""
Assess transformation complexity for effort estimation
"""
function assess_transformation_complexity(sdmx_context, source_context, requirements)
    scores = Dict{String, Float64}()

    # Data size complexity
    scores["data_size"] = calculate_data_size_complexity(source_context)

    # Schema complexity
    scores["schema_complexity"] = calculate_schema_complexity(sdmx_context)

    # Transformation complexity
    scores["transformation_complexity"] = calculate_transformation_complexity(requirements)

    # Excel complexity (if applicable)
    scores["excel_complexity"] = source_context.excel_structure !== nothing ?
                                source_context.excel_structure.complexity_score : 0.0

    # Overall score (weighted average)
    overall_score = (scores["data_size"] * 0.2 +
                    scores["schema_complexity"] * 0.3 +
                    scores["transformation_complexity"] * 0.4 +
                    scores["excel_complexity"] * 0.1)

    # Complexity level
    complexity_level = if overall_score < 2.0
        "simple"
    elseif overall_score < 4.0
        "moderate"
    else
        "complex"
    end

    return (
        overall_score = overall_score,
        complexity_level = complexity_level,
        component_scores = scores,
        estimated_effort_hours = estimate_effort_hours(overall_score),
        key_challenges = identify_key_challenges(requirements)
    )
end

# =================== ENHANCED TRANSFORMATION SCRIPT GENERATION ===================

"""
    generate_enhanced_transformation_script(context::TransformationContext;
                                           provider::Symbol=:ollama,
                                           model::String="",
                                           style::Symbol=:tidier) -> String

Generate comprehensive transformation script using rich metadata context.
"""
function generate_enhanced_transformation_script(context::TransformationContext;
                                                provider::Symbol=:ollama,
                                                model::String="",
                                                style::Symbol=:tidier)
    @assert !isempty(context.transformation_requirements) "Context must have transformation requirements"

    # Build comprehensive context without actual data
    rich_context = build_rich_llm_context(context)
    @assert length(rich_context) > 500 "Context must be substantial"

    # Create specific instructions for each transformation type
    instructions = create_detailed_instructions(context, style)
    @assert !isempty(instructions) "Must have transformation instructions"

    # Generate the script using the enhanced LLM integration
    script_result = sdmx_aigenerate(build_script_generation_prompt(rich_context, instructions, style);
                                   provider=provider, model=model)
    @assert !isempty(script_result.content) "Generated script cannot be empty"

    return script_result.content
end

"""
Build comprehensive LLM context from transformation context
"""
function build_rich_llm_context(context::TransformationContext)
    sdmx_ctx = context.sdmx_context
    source_ctx = context.source_context

    context_parts = [
        "# SDMX DATAFLOW INFORMATION",
        "Dataflow: $(sdmx_ctx.dataflow_info.id) - $(sdmx_ctx.dataflow_info.name)",
        "Agency: $(sdmx_ctx.dataflow_info.agency)",
        "",
        "## Required SDMX Columns",
        join(["- $(col)" for col in sdmx_ctx.required_columns], "\n"),
        "",
        "## SDMX Dimensions with Codelists"
    ]

    # Add detailed codelist information
    for row in eachrow(sdmx_ctx.dimensions)
        dim_id = row.dimension_id
        codelist_id = row.codelist_id

        if haskey(sdmx_ctx.codelist_summary, codelist_id)
            cl_info = sdmx_ctx.codelist_summary[codelist_id]
            codes_sample = join(cl_info.sample_codes, ", ")

            push!(context_parts, "- **$(dim_id)** → Codelist: $(codelist_id)")
            push!(context_parts, "  Total codes: $(cl_info.total_codes)")
            push!(context_parts, "  Sample codes: $(codes_sample)")
            push!(context_parts, "  Hierarchical: $(cl_info.has_hierarchy)")

            # Add available codes if we have them
            if haskey(sdmx_ctx.available_codes, dim_id)
                available = sdmx_ctx.available_codes[dim_id]
                available_sample = join(first(available, min(5, length(available))), ", ")
                push!(context_parts, "  Available in data: $(length(available)) codes (sample: $(available_sample))")
            end
            push!(context_parts, "")
        end
    end

    # Add source data structure
    append!(context_parts, [
        "## SOURCE DATA STRUCTURE",
        "File: $(source_ctx.file_info.path)",
        "Size: $(source_ctx.file_info.size_mb) MB",
        "Dimensions: $(source_ctx.source_profile.row_count) rows × $(source_ctx.source_profile.column_count) columns",
        ""
    ])

    # Add column analysis
    push!(context_parts, "### Source Columns Analysis")
    for col in source_ctx.source_profile.columns
        pattern = source_ctx.column_patterns[col.name]
        col_desc = "- **$(col.name)**: $(pattern.data_type)"

        if pattern.is_temporal
            col_desc *= " (Temporal: $(pattern.temporal_info.format))"
        elseif pattern.is_categorical
            col_desc *= " (Categorical: $(pattern.categorical_info.categories) categories)"
        elseif pattern.is_numeric
            range_info = pattern.numeric_info.range
            col_desc *= " (Numeric: $(range_info[1]) to $(range_info[2]))"
        end

        if pattern.missing_ratio > 0
            col_desc *= " [$(round(pattern.missing_ratio*100, digits=1))% missing]"
        end

        push!(context_parts, col_desc)
    end

    # Add mapping suggestions
    append!(context_parts, ["", "## AUTOMATED MAPPING SUGGESTIONS"])
    for row in eachrow(context.mapping_suggestions)
        confidence_pct = round(row.confidence * 100, digits=1)
        mapping_line = "- $(row.source_column) → $(row.target_column) ($(confidence_pct)% confidence)"
        if !isempty(row.transformation_needed) && row.transformation_needed != "none"
            mapping_line *= " [Needs: $(row.transformation_needed)]"
        end
        if !isnothing(row.codelist_info)
            mapping_line *= " [Codelist: $(row.codelist_info)]"
        end
        push!(context_parts, mapping_line)
    end

    # Add transformation requirements
    append!(context_parts, ["", "## TRANSFORMATION REQUIREMENTS"])
    for req in context.transformation_requirements
        push!(context_parts, "- **$(req.type)** ($(req.priority) priority): $(req.instructions)")
    end

    # Add complexity assessment
    complexity = context.complexity_assessment
    append!(context_parts, [
        "",
        "## COMPLEXITY ASSESSMENT",
        "Overall Level: $(complexity.complexity_level)",
        "Score: $(round(complexity.overall_score, digits=2))/5.0",
        "Estimated Effort: $(complexity.estimated_effort_hours) hours",
        "Key Challenges: $(join(complexity.key_challenges, ", "))"
    ])

    return join(context_parts, "\n")
end

# Helper functions for complexity calculation and instruction generation would be implemented here
# (calculate_data_size_complexity, calculate_schema_complexity, etc.)

function calculate_data_size_complexity(source_context)
    rows = source_context.source_profile.row_count
    cols = source_context.source_profile.column_count

    # Simple complexity scoring based on data size
    size_score = log10(rows * cols) / 2
    return min(size_score, 5.0)  # Cap at 5.0
end

function calculate_schema_complexity(sdmx_context)
    # Score based on number of dimensions, codelists, and hierarchies
    dims_score = nrow(sdmx_context.dimensions) * 0.3
    codelists_score = length(sdmx_context.codelist_summary) * 0.2
    hierarchy_score = sum(cl.has_hierarchy ? 1.0 : 0.0 for cl in values(sdmx_context.codelist_summary)) * 0.5

    return min(dims_score + codelists_score + hierarchy_score, 5.0)
end

function calculate_transformation_complexity(requirements)
    # Score based on transformation types and priorities
    score = 0.0
    for req in requirements
        priority_weight = req.priority == "high" ? 1.0 : req.priority == "medium" ? 0.6 : 0.3
        type_weight = req.type == "pivoting" ? 2.0 : req.type == "code_mapping" ? 1.5 : 1.0
        score += priority_weight * type_weight
    end
    return min(score, 5.0)
end

function estimate_effort_hours(overall_score)
    # Simple effort estimation based on complexity score
    return round(overall_score * 2 + 1, digits=1)  # 1-11 hours range
end

function identify_key_challenges(requirements)
    challenges = String[]

    for req in requirements
        if req.type == "pivoting"
            push!(challenges, "Data reshaping required")
        elseif req.type == "code_mapping" && haskey(req, :codelist_details)
            cl_info = req.codelist_details
            if cl_info.total_codes > 50
                push!(challenges, "Large codelist mapping ($(cl_info.total_codes) codes)")
            end
            if cl_info.has_hierarchy
                push!(challenges, "Hierarchical code mapping")
            end
        elseif req.type == "range_selection"
            push!(challenges, "Excel range selection")
        end
    end

    return unique(challenges)
end

function create_detailed_instructions(context, style)
    # This would generate specific, detailed instructions for the LLM
    # based on the transformation requirements and style preferences
    return "Detailed transformation instructions based on context analysis"
end

function build_script_generation_prompt(rich_context, instructions, style)
    style_instruction = if style == :tidier
        "Use Tidier.jl syntax (@select, @mutate, @filter, @pivot_longer, etc.)"
    elseif style == :dataframes
        "Use DataFrames.jl syntax (select, transform, filter, etc.)"
    else
        "Use a mix of DataFrames.jl and Tidier.jl as appropriate"
    end

    return """
    Generate a comprehensive Julia transformation script for SDMX compliance based on this detailed analysis:

    $rich_context

    ## SCRIPT REQUIREMENTS
    1. $style_instruction
    2. Explicitly use anonymize_source_data(...; target_schema=...) before any LLM-oriented summaries or inspections
    3. Include comprehensive code mapping using the codelist information provided
    4. Handle all identified transformation requirements (pivoting, range selection, etc.)
    5. Add robust validation using @assert statements for data quality checks
    6. Include detailed comments explaining each transformation step
    7. Handle missing data appropriately
    8. Ensure output strictly conforms to SDMX schema requirements
    9. Include unit tests for key transformations

    Generate complete, executable Julia code that can be run independently.
    """
end
