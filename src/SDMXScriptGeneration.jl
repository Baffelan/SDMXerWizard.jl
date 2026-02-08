"""
Tidier.jl Script Generation System for SDMXer.jl

This module generates executable Julia transformation scripts using:
- Template-based code generation with customizable patterns
- Context-aware LLM prompts with comprehensive data analysis
- Tidier.jl best practices and idiomatic patterns
- Scaffolding code that users can extend and refine
- Validation and error handling integration
"""

# Dependencies loaded at package level


"""
    TransformationStep

Represents a single step in a data transformation pipeline for SDMX processing.

This struct encapsulates all information needed to generate a single transformation
operation in a Tidier.jl script, including the operation type, validation checks,
and descriptive comments for the generated code.

# Fields
- `step_name::String`: Descriptive name for this transformation step
- `operation_type::String`: Type of operation ("read", "mutate", "pivot", "filter", "validate", "write")
- `tidier_function::String`: Tidier.jl function to use ("@mutate", "@pivot_longer", etc.)
- `description::String`: Human-readable description of what this step does
- `source_columns::Vector{String}`: Input columns required for this step
- `target_columns::Vector{String}`: Output columns produced by this step
- `transformation_logic::String`: The actual transformation code/expression
- `validation_checks::Vector{String}`: Validation checks to include
- `comments::Vector{String}`: Additional comments for code documentation

# Examples
```julia
step = TransformationStep(
    "Recode Country Values",
    "mutate",
    "@mutate",
    "Maps source country names to SDMX GEO_PICT codes",
    ["country_name"],
    ["GEO_PICT"],
    "GEO_PICT = recode(country_name, \"Tonga\" => \"TO\", \"Fiji\" => \"FJ\")",
    ["@assert all(.!ismissing(GEO_PICT))"],
    ["# Country name standardization for SDMX compliance"]
)
```

# See also
[`ScriptTemplate`](@ref), [`GeneratedScript`](@ref), [`build_transformation_steps`](@ref)
"""
struct TransformationStep
    step_name::String
    operation_type::String  # "read", "mutate", "pivot", "filter", "validate", "write"
    tidier_function::String # "@mutate", "@pivot_longer", etc.
    description::String
    source_columns::Vector{String}
    target_columns::Vector{String}
    transformation_logic::String
    validation_checks::Vector{String}
    comments::Vector{String}
end

"""
    ScriptTemplate

Template for generating specific types of transformation scripts.
"""
struct ScriptTemplate
    template_name::String
    description::String
    use_cases::Vector{String}
    required_packages::Vector{String}
    template_sections::Dict{String, String}
    placeholder_patterns::Dict{String, String}
    example_code::String
end

"""
    GeneratedScript

Complete generated transformation script with metadata.
"""
struct GeneratedScript
    script_name::String
    source_file::String
    target_schema::String
    generated_code::String
    transformation_steps::Vector{TransformationStep}
    estimated_complexity::Float64
    validation_notes::Vector{String}
    user_guidance::Vector{String}
    template_used::String
    generation_timestamp::String
end

"""
    ScriptGenerator

Main generator for creating transformation scripts using the new provider-based approach.
"""
mutable struct ScriptGenerator
    provider::Symbol  # :ollama, :openai, etc.
    model::String
    templates::Dict{String, ScriptTemplate}
    default_template::String
    include_validation::Bool
    include_comments::Bool
    tidier_style::String  # "pipes", "functions", "mixed"
    error_handling_level::String  # "basic", "comprehensive", "production"
end

# =================== CALLABLE STRUCT INTERFACE ===================

"""
    (generator::ScriptGenerator)(profile, schema, mappings; kwargs...) -> GeneratedScript

Make ScriptGenerator callable as a function for intuitive usage:

# Examples
```julia
generator = create_script_generator(llm_config)
script = generator(profile, schema, mappings)  # Instead of generate_transformation_script(generator, ...)

# With custom options
script = generator(profile, schema, mappings; template_name="pivot_transformation")
```
"""
function (generator::ScriptGenerator)(profile::SourceDataProfile,
                                    schema::DataflowSchema,
                                    mappings::AdvancedMappingResult,
                                    excel_analysis::Union{ExcelStructureAnalysis, Nothing}=nothing;
                                    template_name::String="",
                                    custom_instructions::String="")
    return generate_transformation_script(generator, profile, schema, mappings, excel_analysis;
                                        template_name=template_name,
                                        custom_instructions=custom_instructions)
end

# Overload for DataSource input (automatically profiles the data)
function (generator::ScriptGenerator)(source::DataSource,
                                    schema::DataflowSchema,
                                    mappings::AdvancedMappingResult,
                                    excel_analysis::Union{ExcelStructureAnalysis, Nothing}=nothing;
                                    kwargs...)
    data = read_data(source)
    source_info_dict = source_info(source)
    filename = get(source_info_dict, :path, get(source_info_dict, :name, "data_source"))
    profile = profile_source_data(data, filename)

    return generator(profile, schema, mappings, excel_analysis; kwargs...)
end

"""
    create_script_generator(llm_config::LLMConfig;
                           include_validation=true,
                           include_comments=true,
                           tidier_style="pipes",
                           error_handling_level="comprehensive") -> ScriptGenerator

Creates a script generator with specified configuration.
"""
function create_script_generator(provider::Symbol=:ollama, model::String="";
                                include_validation=true,
                                include_comments=true,
                                tidier_style="pipes",
                                error_handling_level="comprehensive")

    generator = ScriptGenerator(
        provider,
        model,
        Dict{String, ScriptTemplate}(),
        "standard_transformation",
        include_validation,
        include_comments,
        tidier_style,
        error_handling_level
    )

    # Load default templates
    load_default_templates!(generator)

    return generator
end

"""
    load_default_templates!(generator::ScriptGenerator)

Loads default script templates into the generator.
"""
function load_default_templates!(generator::ScriptGenerator)
    # Standard transformation template
    standard_template = create_standard_template()
    generator.templates["standard_transformation"] = standard_template

    # Pivot-heavy template
    pivot_template = create_pivot_template()
    generator.templates["pivot_transformation"] = pivot_template

    # Multi-sheet Excel template
    excel_template = create_excel_template()
    generator.templates["excel_multi_sheet"] = excel_template

    # Simple CSV template
    csv_template = create_csv_template()
    generator.templates["simple_csv"] = csv_template
end

"""
    create_standard_template() -> ScriptTemplate

Creates the standard transformation template.
"""
function create_standard_template()
    sections = Dict{String, String}(
        "header" => """
# SDMX Data Transformation Script
# Generated by SDMXer.jl on {{TIMESTAMP}}
# Source: {{SOURCE_FILE}}
# Target: {{TARGET_SCHEMA}}

using DataFrames, CSV, XLSX, Tidier, Statistics, Dates
""",

        "data_loading" => """
# === DATA LOADING ===
println("Loading source data...")
{{DATA_LOADING_CODE}}
println("Loaded \$(nrow(source_data)) rows and \$(ncol(source_data)) columns")
""",

        "data_exploration" => """
# === DATA EXPLORATION ===
println("Source data structure:")
println(describe(source_data))
println("\\nFirst few rows:")
println(first(source_data, 3))
""",

        "transformations" => """
# === DATA TRANSFORMATIONS ===
transformed_data = source_data |>
{{TRANSFORMATION_PIPELINE}}
""",

        "validation" => """
# === DATA VALIDATION ===
println("\\nValidation checks:")
{{VALIDATION_CHECKS}}
""",

        "output" => """
# === DATA OUTPUT ===
println("Writing SDMX-CSV output...")
{{OUTPUT_CODE}}
println("Transformation completed successfully!")
"""
    )

    placeholders = Dict{String, String}(
        "{{TIMESTAMP}}" => "generation_timestamp",
        "{{SOURCE_FILE}}" => "source_file",
        "{{TARGET_SCHEMA}}" => "target_schema",
        "{{DATA_LOADING_CODE}}" => "data_loading_logic",
        "{{TRANSFORMATION_PIPELINE}}" => "transformation_steps",
        "{{VALIDATION_CHECKS}}" => "validation_logic",
        "{{OUTPUT_CODE}}" => "output_logic"
    )

    example_code = """
# Example usage:
source_data = CSV.read("data.csv", DataFrame)
transformed_data = source_data |>
    @mutate(GEO_PICT = recode(country, "FJ" => "FJ", "VU" => "VU")) |>
    @mutate(TIME_PERIOD = string(year)) |>
    @select(GEO_PICT, TIME_PERIOD, OBS_VALUE = value)
"""

    return ScriptTemplate(
        "standard_transformation",
        "Standard SDMX transformation with validation and error handling",
        ["CSV to SDMX-CSV", "Simple data transformations", "Column mapping"],
        ["DataFrames", "CSV", "Tidier", "Statistics"],
        sections,
        placeholders,
        example_code
    )
end

"""
    create_pivot_template() -> ScriptTemplate

Creates template for pivot-heavy transformations.
"""
function create_pivot_template()
    sections = Dict{String, String}(
        "header" => """
# SDMX Pivot Transformation Script
# Generated by SDMXer.jl on {{TIMESTAMP}}
# Handles wide-to-long data pivoting

using DataFrames, CSV, XLSX, Tidier, Statistics, Dates
""",

        "data_loading" => """
# === DATA LOADING ===
{{DATA_LOADING_CODE}}
""",

        "pivot_analysis" => """
# === PIVOT ANALYSIS ===
# Detected time columns: {{TIME_COLUMNS}}
# Pivot strategy: {{PIVOT_STRATEGY}}
""",

        "transformations" => """
# === PIVOT TRANSFORMATIONS ===
transformed_data = source_data |>
{{PIVOT_TRANSFORMATION}}
{{POST_PIVOT_TRANSFORMATIONS}}
""",

        "validation" => """
# === VALIDATION ===
{{VALIDATION_CHECKS}}
""",

        "output" => """
# === OUTPUT ===
{{OUTPUT_CODE}}
"""
    )

    placeholders = Dict{String, String}(
        "{{TIME_COLUMNS}}" => "time_columns",
        "{{PIVOT_STRATEGY}}" => "pivot_strategy",
        "{{PIVOT_TRANSFORMATION}}" => "pivot_logic",
        "{{POST_PIVOT_TRANSFORMATIONS}}" => "post_pivot_steps"
    )

    return ScriptTemplate(
        "pivot_transformation",
        "Template for data requiring pivot operations",
        ["Wide-to-long transformation", "Time series data", "Multi-column pivoting"],
        ["DataFrames", "CSV", "XLSX", "Tidier"],
        sections,
        placeholders,
        ""
    )
end

"""
    create_excel_template() -> ScriptTemplate

Creates template for Excel multi-sheet processing.
"""
function create_excel_template()
    sections = Dict{String, String}(
        "header" => """
# SDMX Excel Multi-Sheet Transformation
# Generated by SDMXer.jl on {{TIMESTAMP}}

using DataFrames, XLSX, Tidier, Statistics
""",

        "excel_analysis" => """
# === EXCEL ANALYSIS ===
# Detected sheets: {{SHEET_NAMES}}
# Primary data sheet: {{PRIMARY_SHEET}}
# Metadata extraction: {{METADATA_INFO}}
""",

        "data_loading" => """
# === DATA LOADING ===
{{EXCEL_LOADING_CODE}}
""",

        "metadata_extraction" => """
# === METADATA EXTRACTION ===
{{METADATA_EXTRACTION_CODE}}
""",

        "transformations" => """
# === TRANSFORMATIONS ===
{{TRANSFORMATION_PIPELINE}}
"""
    )

    return ScriptTemplate(
        "excel_multi_sheet",
        "Template for Excel files with multiple sheets and metadata",
        ["Excel multi-sheet", "Metadata extraction", "Complex Excel structures"],
        ["DataFrames", "XLSX", "Tidier"],
        sections,
        Dict{String, String}(),
        ""
    )
end

"""
    create_csv_template() -> ScriptTemplate

Creates simple CSV template.
"""
function create_csv_template()
    sections = Dict{String, String}(
        "header" => """
# Simple SDMX CSV Transformation
# Generated by SDMXer.jl

using DataFrames, CSV, Tidier
""",

        "transformations" => """
# Load and transform data
source_data = CSV.read("{{SOURCE_FILE}}", DataFrame)

transformed_data = source_data |>
{{TRANSFORMATION_PIPELINE}}

# Save result
CSV.write("sdmx_output.csv", transformed_data)
"""
    )

    return ScriptTemplate(
        "simple_csv",
        "Simple CSV transformation template",
        ["Basic CSV processing", "Quick transformations"],
        ["DataFrames", "CSV", "Tidier"],
        sections,
        Dict{String, String}(),
        ""
    )
end

"""
    generate_transformation_script(generator::ScriptGenerator,
                                 source_profile::SourceDataProfile,
                                 target_schema::DataflowSchema,
                                 mapping_result::AdvancedMappingResult,
                                 excel_analysis::Union{ExcelStructureAnalysis, Nothing} = nothing;
                                 template_name::String = "",
                                 custom_instructions::String = "") -> GeneratedScript

Generates a complete transformation script using LLM and templates.
"""
function generate_transformation_script(generator::ScriptGenerator,
                                       source_profile::SourceDataProfile,
                                       target_schema::DataflowSchema,
                                       mapping_result::AdvancedMappingResult,
                                       excel_analysis::Union{ExcelStructureAnalysis, Nothing} = nothing;
                                       template_name::String = "",
                                       custom_instructions::String = "")

    # Select appropriate template
    selected_template = select_template(generator, source_profile, excel_analysis, template_name)

    # Build transformation steps
    transformation_steps = build_transformation_steps(mapping_result, source_profile, target_schema)

    # Create comprehensive prompt for LLM
    prompt = create_comprehensive_script_prompt(
        generator, selected_template, source_profile, target_schema,
        mapping_result, transformation_steps, excel_analysis, custom_instructions
    )

    # Generate script using LLM
    generated_code = sdmx_aigenerate(
        prompt.user_prompt;
        provider=generator.provider,
        model=generator.model,
        system_prompt=prompt.system_prompt
    )

    # Post-process and validate generated code
    processed_code = post_process_generated_code(generated_code, selected_template, generator)

    # Create validation notes and user guidance
    validation_notes, user_guidance = create_script_guidance(
        mapping_result, transformation_steps, source_profile, target_schema
    )

    # Calculate estimated complexity
    complexity = calculate_script_complexity(transformation_steps, mapping_result)

    return GeneratedScript(
        "sdmx_transformation_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        source_profile.file_path,
        "$(target_schema.dataflow_info.agency):$(target_schema.dataflow_info.id)",
        processed_code,
        transformation_steps,
        complexity,
        validation_notes,
        user_guidance,
        selected_template.template_name,
        string(now())
    )
end

"""
    select_template(generator::ScriptGenerator,
                   source_profile::SourceDataProfile,
                   excel_analysis::Union{ExcelStructureAnalysis, Nothing},
                   template_name::String) -> ScriptTemplate

Selects the most appropriate template based on data characteristics.
"""
function select_template(generator::ScriptGenerator,
                        source_profile::SourceDataProfile,
                        excel_analysis::Union{ExcelStructureAnalysis, Nothing},
                        template_name::String)

    # Use specified template if provided
    if !isempty(template_name) && haskey(generator.templates, template_name)
        return generator.templates[template_name]
    end

    # Auto-select based on data characteristics
    if excel_analysis !== nothing
        if length(excel_analysis.sheets) > 1
            return generator.templates["excel_multi_sheet"]
        elseif excel_analysis.pivoting_detected
            return generator.templates["pivot_transformation"]
        end
    end

    # Check if pivoting is needed based on source profile
    if length(source_profile.suggested_time_columns) > 0 && source_profile.column_count > 5
        return generator.templates["pivot_transformation"]
    end

    # Default to simple CSV for basic cases
    if source_profile.file_type == "csv" && source_profile.column_count <= 8
        return generator.templates["simple_csv"]
    end

    # Default to standard transformation
    return generator.templates[generator.default_template]
end

"""
    build_transformation_steps(mapping_result::AdvancedMappingResult,
                              source_profile::SourceDataProfile,
                              target_schema::DataflowSchema) -> Vector{TransformationStep}

Builds detailed transformation steps from mapping results.
"""
function build_transformation_steps(mapping_result::AdvancedMappingResult,
                                   source_profile::SourceDataProfile,
                                   target_schema::DataflowSchema)

    steps = Vector{TransformationStep}()

    # Step 1: Data loading
    push!(steps, TransformationStep(
        "data_loading",
        "read",
        "CSV.read / XLSX.readtable",
        "Load source data from $(source_profile.file_type) file",
        String[],
        String[],
        get_loading_code(source_profile),
        String[],
        ["Load the source data file with appropriate function"]
    ))

    # Step 2: Data exploration (optional)
    if mapping_result.quality_score < 0.8
        push!(steps, TransformationStep(
            "data_exploration",
            "explore",
            "describe / first",
            "Explore data structure and identify issues",
            collect(String[col.name for col in source_profile.columns]),
            String[],
            "describe(source_data)",
            String[],
            ["Review data structure", "Check for data quality issues", "Understand value distributions"]
        ))
    end

    # Step 3: Column mappings and transformations
    for mapping in mapping_result.mappings
        step_name = "map_$(mapping.source_column)_to_$(mapping.target_column)"

        if mapping.suggested_transformation !== nothing
            # Complex transformation needed
            push!(steps, TransformationStep(
                step_name,
                "mutate",
                "@mutate",
                "Transform $(mapping.source_column) to $(mapping.target_column)",
                [mapping.source_column],
                [mapping.target_column],
                generate_transformation_logic(mapping),
                ["Check for missing values", "Validate transformation results"],
                ["$(mapping.confidence_level) confidence mapping", "$(mapping.match_type) match"]
            ))
        else
            # Simple column rename/select
            push!(steps, TransformationStep(
                step_name,
                "select",
                "@select",
                "Map $(mapping.source_column) to $(mapping.target_column)",
                [mapping.source_column],
                [mapping.target_column],
                "$(mapping.target_column) = $(mapping.source_column)",
                String[],
                ["Direct column mapping"]
            ))
        end
    end

    # Step 4: Handle missing required columns
    required_cols = get_required_columns(target_schema)
    mapped_targets = Set([m.target_column for m in mapping_result.mappings])
    missing_required = setdiff(Set(required_cols), mapped_targets)

    if !isempty(missing_required)
        push!(steps, TransformationStep(
            "handle_missing_columns",
            "mutate",
            "@mutate",
            "Handle missing required columns",
            String[],
            collect(missing_required),
            "# TODO: Add logic for missing columns: $(join(missing_required, ", "))",
            ["Ensure all required columns are present"],
            ["Manual intervention needed for unmapped required columns"]
        ))
    end

    # Step 5: Data validation
    if any(m.confidence_level <= MEDIUM for m in mapping_result.mappings)
        push!(steps, TransformationStep(
            "data_validation",
            "validate",
            "custom validation",
            "Validate transformation results",
            String[],
            String[],
            create_validation_logic(mapping_result, target_schema),
            ["Check data completeness", "Validate against SDMX requirements"],
            ["Review low-confidence mappings", "Check for data quality issues"]
        ))
    end

    # Step 6: Output
    push!(steps, TransformationStep(
        "data_output",
        "write",
        "CSV.write",
        "Write SDMX-compliant CSV output",
        String[],
        String[],
        "CSV.write(\"sdmx_output.csv\", transformed_data)",
        ["Verify output format compliance"],
        ["Save result as SDMX-CSV format"]
    ))

    return steps
end

"""
    create_comprehensive_script_prompt(generator::ScriptGenerator,
                                      template::ScriptTemplate,
                                      source_profile::SourceDataProfile,
                                      target_schema::DataflowSchema,
                                      mapping_result::AdvancedMappingResult,
                                      transformation_steps::Vector{TransformationStep},
                                      excel_analysis::Union{ExcelStructureAnalysis, Nothing},
                                      custom_instructions::String) -> NamedTuple

Creates comprehensive prompts for LLM script generation.
"""
function create_comprehensive_script_prompt(generator::ScriptGenerator,
                                           template::ScriptTemplate,
                                           source_profile::SourceDataProfile,
                                           target_schema::DataflowSchema,
                                           mapping_result::AdvancedMappingResult,
                                           transformation_steps::Vector{TransformationStep},
                                           excel_analysis::Union{ExcelStructureAnalysis, Nothing},
                                           custom_instructions::String)

    system_prompt = """You are an expert Julia developer specializing in data transformation using Tidier.jl and SDMX standards.

Your task is to generate a working Julia script that transforms source data into SDMX-compliant format. The script should:

1. Use Tidier.jl with $(generator.tidier_style) style for data manipulation
2. Include $(generator.error_handling_level) error handling
3. Follow Julia best practices and be well-documented
4. Generate working code that users can extend and refine
5. Include data validation and quality checks
6. Use the provided template structure as a foundation

Key requirements:
- Prefer @mutate, @select, @filter, @pivot_longer/@pivot_wider from Tidier.jl
- Include informative println statements for progress tracking
- Add TODO comments where manual intervention is needed
- Generate code that handles edge cases gracefully
- Focus on clarity and maintainability over optimization"""

    user_prompt = """Please generate a complete Julia transformation script based on the following analysis:

## Template: $(template.template_name)
$(template.description)

Required packages: $(join(template.required_packages, ", "))

## Source Data Analysis
File: $(source_profile.file_path) ($(source_profile.file_type))
Dimensions: $(source_profile.row_count) rows × $(source_profile.column_count) columns
Data Quality: $(round(source_profile.data_quality_score * 100, digits=1))%

### Source Columns:
"""

    for col in source_profile.columns
        user_prompt *= "- **$(col.name)**: $(col.type)"
        if col.is_temporal
            user_prompt *= " (Time: $(col.temporal_format))"
        elseif col.is_categorical
            user_prompt *= " (Categorical: $(col.unique_count) values)"
        elseif col.numeric_stats !== nothing
            user_prompt *= " (Numeric: $(round(col.numeric_stats.min, digits=1))-$(round(col.numeric_stats.max, digits=1)))"
        end
        user_prompt *= "\n"
    end

    # Add Excel analysis if available
    if excel_analysis !== nothing
        user_prompt *= "\n### Excel Analysis:\n"
        user_prompt *= "- Primary sheet: $(excel_analysis.recommended_sheet)\n"
        user_prompt *= "- Data start row: $(excel_analysis.data_start_row)\n"
        user_prompt *= "- Header row: $(excel_analysis.header_row)\n"
        user_prompt *= "- Pivoting detected: $(excel_analysis.pivot_structure_detected)\n"

        if !isempty(excel_analysis.data_quality_issues)
            user_prompt *= "- Data quality issues: $(join(excel_analysis.data_quality_issues, "; "))\n"
        end

        if !isempty(excel_analysis.recommended_preprocessing)
            user_prompt *= "- Preprocessing: $(join(excel_analysis.recommended_preprocessing, "; "))\n"
        end
    end

    # Add target schema information
    user_prompt *= "\n## Target SDMX Schema\n"
    user_prompt *= "Dataflow: $(target_schema.dataflow_info.id) - $(target_schema.dataflow_info.name)\n"
    user_prompt *= "Agency: $(target_schema.dataflow_info.agency)\n\n"

    user_prompt *= "### Required Columns ($(length(get_required_columns(target_schema)))):\n"
    for col in get_required_columns(target_schema)
        user_prompt *= "- $col\n"
    end

    optional_cols = get_optional_columns(target_schema)
    if !isempty(optional_cols)
        user_prompt *= "\n### Optional Columns ($(length(optional_cols))):\n"
        for col in optional_cols[1:min(5, length(optional_cols))]
            user_prompt *= "- $col\n"
        end
    end

    # Add mapping results
    user_prompt *= "\n## Mapping Analysis\n"
    user_prompt *= "Quality Score: $(round(mapping_result.quality_score, digits=3))\n"
    user_prompt *= "Coverage: $(round(mapping_result.coverage_analysis["required_coverage"], digits=3))\n"
    user_prompt *= "Complexity: $(round(mapping_result.transformation_complexity, digits=3))\n\n"

    user_prompt *= "### Confirmed Mappings:\n"
    for mapping in mapping_result.mappings[1:min(8, length(mapping_result.mappings))]
        user_prompt *= "- **$(mapping.source_column)** → **$(mapping.target_column)** "
        user_prompt *= "($(mapping.confidence_level), $(mapping.match_type))\n"
        if mapping.suggested_transformation !== nothing
            user_prompt *= "  Transformation: $(mapping.suggested_transformation)\n"
        end
    end

    if !isempty(mapping_result.unmapped_source_columns)
        user_prompt *= "\n### Unmapped Source Columns:\n$(join(mapping_result.unmapped_source_columns, ", "))\n"
    end

    if !isempty(mapping_result.unmapped_target_columns)
        user_prompt *= "\n### Missing Target Columns:\n$(join(mapping_result.unmapped_target_columns, ", "))\n"
    end

    # Add transformation steps guidance
    user_prompt *= "\n## Transformation Steps Required:\n"
    for (i, step) in enumerate(transformation_steps)
        user_prompt *= "$(i). **$(step.step_name)**: $(step.description)\n"
        if !isempty(step.comments)
            user_prompt *= "   Notes: $(join(step.comments, "; "))\n"
        end
    end

    # Add recommendations
    if !isempty(mapping_result.recommendations)
        user_prompt *= "\n## Recommendations:\n"
        for rec in mapping_result.recommendations
            user_prompt *= "- $rec\n"
        end
    end

    # Add custom instructions
    if !isempty(custom_instructions)
        user_prompt *= "\n## Custom Instructions:\n$custom_instructions\n"
    end

    # Add template structure
    user_prompt *= "\n## Template Structure:\n"
    user_prompt *= "Use this template as a foundation, adapting as needed:\n\n"

    # Show key template sections
    for (section_name, section_code) in template.template_sections
        user_prompt *= "### $section_name:\n```julia\n$section_code\n```\n\n"
    end

    user_prompt *= """
## Output Requirements:
1. Generate complete, executable Julia code
2. Use Tidier.jl syntax throughout (@mutate, @select, @filter, @pivot_longer, etc.)
3. Include comprehensive comments explaining each transformation
4. Add TODO comments where manual intervention is needed
5. Include progress tracking with println statements
6. Handle edge cases and missing data appropriately
7. Validate the transformation results
8. Save output as SDMX-compliant CSV

Generate the complete script now:"""

    return (system_prompt=system_prompt, user_prompt=user_prompt)
end

"""
    get_loading_code(source_profile::SourceDataProfile) -> String

Generates appropriate data loading code based on file type.
"""
function get_loading_code(source_profile::SourceDataProfile)
    if source_profile.file_type == "csv"
        return """source_data = CSV.read("$(source_profile.file_path)", DataFrame)"""
    elseif source_profile.file_type in ["xlsx", "xls"]
        return """source_data = XLSX.readtable("$(source_profile.file_path)", 1) |> DataFrame
# Note: Adjust sheet number/name as needed"""
    else
        return "# TODO: Add appropriate loading code for $(source_profile.file_type) files"
    end
end

"""
    generate_transformation_logic(mapping::MappingCandidate) -> String

Generates Tidier.jl transformation logic for a specific mapping.
"""
function generate_transformation_logic(mapping::MappingCandidate)
    if mapping.suggested_transformation !== nothing
        return mapping.suggested_transformation
    end

    # Generate basic transformation based on mapping type
    if mapping.match_type == "exact"
        return "$(mapping.target_column) = $(mapping.source_column)"
    elseif mapping.match_type == "fuzzy" || mapping.match_type == "name_fuzzy"
        return "$(mapping.target_column) = $(mapping.source_column)  # Review fuzzy match"
    else
        return "$(mapping.target_column) = $(mapping.source_column)  # TODO: Verify mapping"
    end
end

"""
    create_validation_logic(mapping_result::AdvancedMappingResult,
                           target_schema::DataflowSchema) -> String

Creates validation logic for the transformation.
"""
function create_validation_logic(mapping_result::AdvancedMappingResult,
                                target_schema::DataflowSchema)
    validation_code = """
# Check required columns are present
required_cols = $(get_required_columns(target_schema))
missing_cols = setdiff(required_cols, names(transformed_data))
if !isempty(missing_cols)
    @warn "Missing required columns: \$missing_cols"
end

# Check for missing values in key columns
for col in names(transformed_data)
    missing_count = sum(ismissing.(transformed_data[!, col]))
    if missing_count > 0
        println("Column '\$col' has \$missing_count missing values")
    end
end

println("Validation completed.")"""

    return validation_code
end

"""
    post_process_generated_code(code::String, template::ScriptTemplate,
                               generator::ScriptGenerator) -> String

Post-processes generated code to ensure quality and consistency.
"""
function post_process_generated_code(code::String, template::ScriptTemplate,
                                   generator::ScriptGenerator)
    processed_code = code

    # Remove any markdown code blocks
    processed_code = replace(processed_code, r"```julia\n?" => "")
    processed_code = replace(processed_code, r"```\n?" => "")

    # Ensure proper imports are included
    if !occursin("using", processed_code)
        processed_code = join(template.required_packages, ", ") * "\n\n" * processed_code
    end

    # Add generation comment if not present
    if !occursin("Generated by SDMXer.jl", processed_code)
        header_comment = "# Generated by SDMXer.jl on $(now())\n# Template: $(template.template_name)\n\n"
        processed_code = header_comment * processed_code
    end

    return processed_code
end

"""
    create_script_guidance(mapping_result::AdvancedMappingResult,
                          transformation_steps::Vector{TransformationStep},
                          source_profile::SourceDataProfile,
                          target_schema::DataflowSchema) -> Tuple{Vector{String}, Vector{String}}

Creates validation notes and user guidance for the generated script.
"""
function create_script_guidance(mapping_result::AdvancedMappingResult,
                               transformation_steps::Vector{TransformationStep},
                               source_profile::SourceDataProfile,
                               target_schema::DataflowSchema)

    validation_notes = String[]
    user_guidance = String[]

    # Validation notes
    if mapping_result.quality_score < 0.7
        push!(validation_notes, "Low mapping quality score ($(round(mapping_result.quality_score, digits=2))) - review transformations carefully")
    end

    if mapping_result.coverage_analysis["required_coverage"] < 0.8
        push!(validation_notes, "Incomplete coverage of required fields ($(round(mapping_result.coverage_analysis["required_coverage"]*100, digits=1))%)")
    end

    low_confidence_mappings = sum([m.confidence_level <= MEDIUM for m in mapping_result.mappings])
    if low_confidence_mappings > 0
        push!(validation_notes, "$low_confidence_mappings mappings have medium or low confidence")
    end

    # User guidance
    push!(user_guidance, "Review and test the generated script before processing production data")
    push!(user_guidance, "Pay special attention to TODO comments - manual intervention may be needed")
    push!(user_guidance, "Validate the output against SDMX requirements for your specific dataflow")

    if source_profile.data_quality_score < 0.9
        push!(user_guidance, "Source data quality is $(round(source_profile.data_quality_score*100, digits=1))% - consider data cleaning")
    end

    if !isempty(mapping_result.unmapped_target_columns)
        push!(user_guidance, "$(length(mapping_result.unmapped_target_columns)) target columns remain unmapped - add logic as needed")
    end

    return validation_notes, user_guidance
end

"""
    calculate_script_complexity(transformation_steps::Vector{TransformationStep},
                                mapping_result::AdvancedMappingResult) -> Float64

Calculates estimated complexity of the generated script.
"""
function calculate_script_complexity(transformation_steps::Vector{TransformationStep},
                                    mapping_result::AdvancedMappingResult)

    complexity = 0.0

    # Base complexity from number of transformation steps
    complexity += length(transformation_steps) * 0.1

    # Complexity from transformation types
    for step in transformation_steps
        if step.operation_type in ["pivot", "validate"]
            complexity += 0.2
        elseif step.operation_type in ["mutate"]
            complexity += 0.1
        end
    end

    # Complexity from mapping quality
    complexity += (1.0 - mapping_result.quality_score) * 0.3

    # Complexity from transformation needs
    complex_transformations = sum([m.suggested_transformation !== nothing for m in mapping_result.mappings])
    complexity += complex_transformations * 0.05

    return min(1.0, complexity)
end

"""
    validate_generated_script(script::GeneratedScript) -> Dict{String, Any}

Validates the generated script for common issues.
"""
function validate_generated_script(script::GeneratedScript)
    validation = Dict{String, Any}(
        "syntax_issues" => String[],
        "missing_elements" => String[],
        "recommendations" => String[],
        "overall_quality" => "unknown"
    )

    code = script.generated_code

    # Check for required imports
    required_packages = ["DataFrames", "Tidier"]
    for package in required_packages
        if !occursin("using $package", code) && !occursin("import $package", code)
            push!(validation["missing_elements"], "Missing import for $package")
        end
    end

    # Check for Tidier.jl usage
    tidier_functions = ["@mutate", "@select", "@filter", "@pivot_longer", "@pivot_wider"]
    tidier_used = any(func -> occursin(func, code), tidier_functions)
    if !tidier_used
        push!(validation["missing_elements"], "No Tidier.jl functions detected")
    end

    # Check for data loading
    if !occursin("CSV.read", code) && !occursin("XLSX.read", code)
        push!(validation["missing_elements"], "No data loading code detected")
    end

    # Check for output
    if !occursin("CSV.write", code) && !occursin("write", code)
        push!(validation["missing_elements"], "No data output code detected")
    end

    # Recommendations
    if script.estimated_complexity > 0.7
        push!(validation["recommendations"], "High complexity script - consider breaking into smaller steps")
    end

    if length(script.validation_notes) > 3
        push!(validation["recommendations"], "Multiple validation concerns - thorough testing recommended")
    end

    # Overall quality assessment
    issues = length(validation["syntax_issues"]) + length(validation["missing_elements"])
    if issues == 0
        validation["overall_quality"] = "good"
    elseif issues <= 2
        validation["overall_quality"] = "acceptable"
    else
        validation["overall_quality"] = "needs_work"
    end

    return validation
end

"""
    preview_script_output(script::GeneratedScript; max_lines::Int = 50) -> String

Creates a preview of the generated script for user review.
"""
function preview_script_output(script::GeneratedScript; max_lines::Int = 50)
    preview = """
=== GENERATED SCRIPT PREVIEW ===
Script: $(script.script_name)
Template: $(script.template_used)
Generated: $(script.generation_timestamp)
Complexity: $(round(script.estimated_complexity, digits=2))

Source: $(script.source_file)
Target: $(script.target_schema)

=== TRANSFORMATION STEPS ($(length(script.transformation_steps))) ===
"""

    for (i, step) in enumerate(script.transformation_steps)
        preview *= "$(i). $(step.step_name): $(step.description)\n"
    end

    if !isempty(script.validation_notes)
        preview *= "\n=== VALIDATION NOTES ===\n"
        for note in script.validation_notes
            preview *= "⚠ $note\n"
        end
    end

    if !isempty(script.user_guidance)
        preview *= "\n=== USER GUIDANCE ===\n"
        for guidance in script.user_guidance
            preview *= "💡 $guidance\n"
        end
    end

    preview *= "\n=== CODE PREVIEW (first $max_lines lines) ===\n"
    code_lines = split(script.generated_code, '\n')
    preview_lines = code_lines[1:min(max_lines, length(code_lines))]
    preview *= join(preview_lines, '\n')

    if length(code_lines) > max_lines
        preview *= "\n... ($(length(code_lines) - max_lines) more lines)"
    end

    return preview
end
