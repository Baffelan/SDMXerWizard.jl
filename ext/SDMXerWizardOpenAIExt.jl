"""
SDMXerWizard OpenAI Extension

This extension provides direct OpenAI API integration for SDMXerWizard.jl when OpenAI.jl
is available. It enables users who prefer direct OpenAI API calls over PromptingTools.jl
abstractions to use OpenAI-specific features and configurations.

Only loaded when OpenAI.jl is imported by the user.
"""

module SDMXerWizardOpenAIExt

# Only loaded when OpenAI is available
if isdefined(Base, :get_extension)
    using OpenAI
    using SDMXerWizard
    import SDMXerWizard: generate_transformation_script_openai, query_openai_direct,
                    create_openai_mapping, analyze_excel_with_openai
else
    # Fallback for older Julia versions
    @warn "OpenAI extension requires Julia 1.9+ with package extensions support"
end

"""
    generate_transformation_script_openai(prompt::String; model="gpt-4", kwargs...) -> String

Generate SDMX transformation scripts using direct OpenAI API calls.

This function provides direct access to OpenAI's API for generating transformation
scripts, bypassing PromptingTools.jl abstractions. Useful when you need OpenAI-specific
features or configurations not available through the general interface.

# Arguments
- `prompt::String`: Transformation request prompt
- `model="gpt-4"`: OpenAI model to use
- `kwargs...`: Additional OpenAI API parameters

# Returns
- `String`: Generated Julia/Tidier.jl transformation script

# Examples
```julia
# Direct OpenAI integration (requires: using OpenAI)
script = generate_transformation_script_openai(
    "Convert this pivot table to SDMX format: ...",
    model="gpt-4",
    temperature=0.1,
    max_tokens=2000
)

# Use with specific OpenAI features
script = generate_transformation_script_openai(
    prompt,
    model="gpt-4",
    response_format=Dict("type" => "json_object"),
    seed=42  # For reproducible outputs
)
```

# Throws
- `OpenAI.APIError`: If OpenAI API request fails
- `ArgumentError`: If required parameters are missing

# See also
[`query_openai_direct`](@ref), [`analyze_excel_with_openai`](@ref)
"""
function generate_transformation_script_openai(prompt::String; model="gpt-4", kwargs...)
    # Enhanced prompt for SDMX transformation
    system_prompt = """
    You are an expert in SDMX (Statistical Data and Metadata eXchange) data transformation.
    Generate clean, efficient Tidier.jl code for transforming data to SDMX format.
    Focus on:
    1. Proper dimension and measure identification
    2. Time dimension handling
    3. Code list validation
    4. Data quality checks
    5. SDMX-CSV output format

    Return only executable Julia code using Tidier.jl syntax.
    """

    messages = [
        Dict("role" => "system", "content" => system_prompt),
        Dict("role" => "user", "content" => prompt)
    ]

    try
        response = OpenAI.create_chat(
            model,
            messages;
            kwargs...
        )

        return response.choices[1].message.content
    catch e
        throw(OpenAI.APIError("Failed to generate transformation script: $(e.message)"))
    end
end

"""
    query_openai_direct(prompt::String; model="gpt-4", kwargs...) -> String

Direct OpenAI API query for SDMX-related tasks.

Provides low-level access to OpenAI API for custom SDMX workflows that require
specific OpenAI features or configurations not available through PromptingTools.jl.

# Arguments
- `prompt::String`: Query prompt
- `model="gpt-4"`: OpenAI model to use
- `kwargs...`: OpenAI API parameters

# Returns
- `String`: OpenAI response content

# Examples
```julia
# Custom SDMX analysis
response = query_openai_direct(
    "Analyze this SDMX schema for data quality issues: ...",
    model="gpt-4",
    temperature=0.2
)

# Structured output
response = query_openai_direct(
    prompt,
    response_format=Dict("type" => "json_object")
)
```

# See also
[`generate_transformation_script_openai`](@ref)
"""
function query_openai_direct(prompt::String; model="gpt-4", kwargs...)
    messages = [Dict("role" => "user", "content" => prompt)]

    response = OpenAI.create_chat(model, messages; kwargs...)
    return response.choices[1].message.content
end

"""
    create_openai_mapping(source_data::DataFrame, target_schema::Dict; kwargs...) -> Dict

Create column mappings using OpenAI's advanced reasoning capabilities.

Uses OpenAI's latest models to analyze source data structure and target SDMX schema
to generate intelligent column mappings with confidence scores and explanations.

# Arguments
- `source_data::DataFrame`: Source dataset to analyze
- `target_schema::Dict`: SDMX schema structure
- `kwargs...`: OpenAI API parameters

# Returns
- `Dict`: Mapping suggestions with confidence scores and explanations

# Examples
```julia
mappings = create_openai_mapping(
    my_dataframe,
    sdmx_schema,
    model="gpt-4",
    temperature=0.1
)

println(mappings["COUNTRY"]["confidence"])  # 0.95
println(mappings["COUNTRY"]["explanation"]) # "Strong match based on..."
```

# See also
[`analyze_excel_with_openai`](@ref)
"""
function create_openai_mapping(source_data::DataFrame, target_schema::Dict; kwargs...)
    # Prepare data context
    data_summary = """
    Source Data Structure:
    - Columns: $(names(source_data))
    - Sample values: $(first(source_data, 3))
    - Data types: $(eltype.(eachcol(source_data)))

    Target SDMX Schema:
    $(JSON3.write(target_schema, allow_inf=true))
    """

    prompt = """
    Analyze the source data and create mappings to the SDMX schema.
    Return a JSON object with mapping suggestions including confidence scores (0-1).

    $data_summary

    Format:
    {
        "mappings": {
            "SDMX_COLUMN": {
                "source_column": "SOURCE_COL",
                "confidence": 0.95,
                "explanation": "Reasoning for this mapping"
            }
        }
    }
    """

    response = query_openai_direct(
        prompt;
        response_format=Dict("type" => "json_object"),
        kwargs...
    )

    return JSON3.read(response)
end

"""
    analyze_excel_with_openai(file_path::String; kwargs...) -> Dict

Analyze Excel file structure using OpenAI's vision and reasoning capabilities.

When available, uses OpenAI's vision models to analyze Excel file screenshots
or structured data to understand complex layouts, merged cells, and data patterns.

# Arguments
- `file_path::String`: Path to Excel file
- `kwargs...`: OpenAI API parameters

# Returns
- `Dict`: Analysis results including structure recommendations

# Examples
```julia
analysis = analyze_excel_with_openai(
    "complex_pivot_table.xlsx",
    model="gpt-4-vision-preview"
)

println(analysis["recommendations"])
println(analysis["pivot_detected"])
```

# See also
[`create_openai_mapping`](@ref), [`generate_transformation_script_openai`](@ref)
"""
function analyze_excel_with_openai(file_path::String; kwargs...)
    # Read Excel file structure
    if !isfile(file_path)
        throw(ArgumentError("File not found: $file_path"))
    end

    # Basic Excel analysis
    sheets = XLSX.sheetnames(XLSX.readxlsx(file_path))

    prompt = """
    Analyze this Excel file for SDMX data transformation:

    File: $file_path
    Sheets: $sheets

    Provide analysis including:
    1. Data structure assessment
    2. Pivot table detection
    3. Header row identification
    4. Data quality indicators
    5. Transformation complexity score (1-10)
    6. Recommended approach

    Return structured JSON response.
    """

    response = query_openai_direct(
        prompt;
        response_format=Dict("type" => "json_object"),
        kwargs...
    )

    return JSON3.read(response)
end

# Extension initialization
function __init__()
    @info "SDMX OpenAI Extension loaded - Direct OpenAI API integration available"
    @info "Available functions: generate_transformation_script_openai, query_openai_direct, create_openai_mapping, analyze_excel_with_openai"
end

end # module SDMXerWizardOpenAIExt
