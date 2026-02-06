"""
PromptingTools.jl Integration for SDMXer.jl

This module provides streamlined AI-powered analysis using PromptingTools.jl:
- Multi-provider LLM support (OpenAI, Anthropic, Ollama, etc.)
- Structured data extraction with type-safe responses
- Prompt templates for SDMX-specific tasks
- Conversation management and cost tracking
- Advanced Excel analysis and transformation script generation
"""

# Dependencies loaded at package level

# =================== ENUMS AND TYPES ===================

"Provider types for LLM configuration"
@enum LLMProvider begin
    OPENAI
    ANTHROPIC 
    OLLAMA
    MISTRAL
    AZURE_OPENAI
    GOOGLE
end

"Script generation styles"
@enum ScriptStyle begin
    DATAFRAMES
    TIDIER
    MIXED
end

"Convert symbols to LLMProvider enum"
function llm_provider(s::Symbol)::LLMProvider
    s === :openai && return OPENAI
    s === :anthropic && return ANTHROPIC
    s === :ollama && return OLLAMA
    s === :mistral && return MISTRAL
    s === :azure_openai && return AZURE_OPENAI
    s === :google && return GOOGLE
    error("Unknown LLM provider: $s")
end

"Convert symbols to ScriptStyle enum"
function script_style_enum(s::Symbol)::ScriptStyle
    s === :dataframes && return DATAFRAMES
    s === :tidier && return TIDIER
    s === :mixed && return MIXED
    error("Unknown script style: $s")
end


# =================== PROVIDER CONFIGURATION ===================

"""
Supported LLM providers with their PromptingTools.jl schemas and capabilities
"""
const SDMX_PROVIDERS = Dict{Symbol, NamedTuple}(
    :openai => (
        schema = PromptingTools.OpenAISchema(),
        supports = [:generation, :extraction, :embedding, :classification],
        default_model = "gpt-4o",
        description = "OpenAI GPT models"
    ),
    :anthropic => (
        schema = PromptingTools.AnthropicSchema(),
        supports = [:generation, :extraction],
        default_model = "claude-3-5-sonnet-20241022",
        description = "Anthropic Claude models"
    ),
    :ollama => (
        schema = PromptingTools.OllamaSchema(),
        supports = [:generation, :embedding],
        default_model = "qwen3:0.6b",
        description = "Local Ollama models"
    ),
    :mistral => (
        schema = PromptingTools.MistralOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "mistral-large",
        description = "Mistral AI models"
    ),
    :groq => (
        schema = PromptingTools.GroqOpenAISchema(),
        supports = [:generation, :extraction],
        default_model = "llama-3.1-70b-versatile",
        description = "Groq fast inference"
    ),
    :together => (
        schema = PromptingTools.TogetherOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "meta-llama/Llama-3-70b-chat-hf",
        description = "Together AI models"
    ),
    :fireworks => (
        schema = PromptingTools.FireworksOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "accounts/fireworks/models/llama-v3-70b-instruct",
        description = "Fireworks AI models"
    ),
    :databricks => (
        schema = PromptingTools.DatabricksOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "databricks-meta-llama-3-70b-instruct",
        description = "Databricks models"
    ),
    :google => (
        schema = PromptingTools.GoogleSchema(),
        supports = [:generation],
        default_model = "gemini-1.5-pro",
        description = "Google Gemini models"
    )
)

# =================== STRUCTURED RESPONSE TYPES ===================

"""
    SDMXMappingResult

Structured result from AI-powered column mapping analysis.
Used with aiextract() for type-safe responses.
"""
struct SDMXMappingResult
    source_columns::Vector{String}
    suggested_mappings::Vector{Dict{String, Any}}
    confidence_scores::Vector{Float64}
    transformation_notes::Vector{String}
    recommended_actions::Vector{String}
end

"""
    SDMXTransformationScript

Structured result from AI-generated transformation script.
Used with aiextract() for type-safe script generation.
"""
struct SDMXTransformationScript
    script_type::String  # "tidier", "dataframes", "mixed"
    script_content::String
    required_packages::Vector{String}
    validation_steps::Vector{String}
    known_limitations::Vector{String}
    estimated_complexity::String  # "simple", "moderate", "complex"
end

"""
    ExcelStructureAnalysis

Structured result from AI-powered Excel analysis.
Used with aiextract() for comprehensive Excel understanding.
"""
struct ExcelStructureAnalysis
    recommended_sheet::String
    data_start_row::Int
    data_end_row::Int
    header_row::Int
    column_descriptions::Vector{Dict{String, String}}
    pivot_structure_detected::Bool
    data_quality_issues::Vector{String}
    recommended_preprocessing::Vector{String}
end

# =================== SETUP AND CONFIGURATION ===================

"""
    setup_sdmx_llm(provider::Union{Symbol, LLMProvider}=:openai; model::String="", env_file::Union{String, Nothing}=nothing, kwargs...)

Sets up PromptingTools.jl for SDMX tasks with provider-specific configurations.
Supports all PromptingTools.jl providers with sensible defaults.
Optionally loads API keys from a YAML .env file.

# Arguments
- `provider`: LLM provider (:openai, :anthropic, :ollama, :google, etc.)
- `model`: Model name (uses provider defaults if empty)
- `env_file`: Path to YAML .env file containing API keys (optional)

# Important Note for Google AI
⚠️ **Google AI Requirement**: The `GOOGLE_API_KEY` environment variable must be set 
BEFORE importing SDMXerWizard. The `.env` file loading will set the variable for future 
imports but cannot affect already-loaded GoogleGenAI.jl.

```julia
# Correct usage for Google:
ENV["GOOGLE_API_KEY"] = "your-api-key"
using SDMXerWizard
setup_sdmx_llm(:google)
```

# Examples
```julia
# OpenAI (default) - using symbols (recommended)
setup_sdmx_llm(:openai)  # Uses gpt-4o by default

# With API keys from .env file (works for most providers)
setup_sdmx_llm(:openai, env_file=".env")

# Google Gemini (requires ENV var before import)
ENV["GOOGLE_API_KEY"] = "your-key"
using SDMXerWizard
setup_sdmx_llm(:google, model="gemini-1.5-flash")

# Anthropic
setup_sdmx_llm(:anthropic, model="claude-3-sonnet-20240229")

# Local Ollama
setup_sdmx_llm(:ollama, model="llama3")
```

# .env File Format
The .env file should be in YAML format:
```yaml
GOOGLE_API_KEY: "your-google-api-key"
OPENAI_API_KEY: "your-openai-api-key"
ANTHROPIC_API_KEY: "your-anthropic-api-key"
```
"""
function setup_sdmx_llm(provider::Union{Symbol, LLMProvider}=:openai; model::String="", env_file::Union{String, Nothing}=nothing, kwargs...)
    # Load API keys from .env file if specified
    if env_file !== nothing
        _load_api_keys_from_env(env_file)
    end
    
    # Convert symbol to enum if needed
    provider_enum = isa(provider, Symbol) ? llm_provider(provider) : provider
    
    # Special warning for Google provider
    if provider_enum === GOOGLE && !haskey(ENV, "GOOGLE_API_KEY")
        @warn "GOOGLE_API_KEY not found in environment!" * 
              "\n  Google AI requires the API key to be set BEFORE importing SDMXerWizard." *
              "\n  Please use the init_sdmx_google.jl helper script or set ENV[\"GOOGLE_API_KEY\"] before import." *
              "\n  See the README for detailed instructions."
    end
    
    # Set provider-specific defaults optimized for SDMX tasks
    model = _get_default_model(provider_enum, model)
    
    # Configure PromptingTools.jl model alias
    !isempty(model) && (PromptingTools.MODEL_ALIASES["sdmx_model"] = model)
    
    @info "SDMX LLM configured: provider=" * string(provider_enum) * ", model=" * model
    return provider_enum
end

# Load API keys from YAML .env file
function _load_api_keys_from_env(env_file::String)
    @assert isfile(env_file) "Environment file not found at: " * env_file
    
    config = YAML.load_file(env_file)
    
    # Set environment variables for PromptingTools.jl
    for (key, value) in config
        if endswith(key, "_API_KEY") || endswith(key, "_ENDPOINT")
            ENV[key] = string(value)
        end
    end
    
    @info "API keys loaded from " * env_file
end

# Get default model for each provider
function _get_default_model(provider::LLMProvider, model::String)::String
    !isempty(model) && return model
    
    provider === OPENAI && return "gpt-4o"
    provider === ANTHROPIC && return "claude-3-5-sonnet-20241022"
    provider === OLLAMA && return "llama3.1:8b"
    provider === MISTRAL && return "mistral-large"
    provider === AZURE_OPENAI && return "gpt-4o"
    provider === GOOGLE && return "gemini-2.5-flash"
    
    return ""
end

# =================== DATA PREPARATION FUNCTIONS ===================

"""
    prepare_data_preview(data::DataFrame) -> NamedTuple

Prepare data preview for AI analysis using functional approach.
"""
function prepare_data_preview(data::DataFrame)
    cols = names(data)
    return (
        columns = cols,
        types = eltype.(eachcol(data)),
        sample_values = NamedTuple(Symbol(col) => _sample_unique(data[!, col]) for col in cols),
        dimensions = size(data)
    )
end

"""
    prepare_schema_context(schema::DataflowSchema) -> NamedTuple

Extract SDMX schema information for AI context.
"""
function prepare_schema_context(schema::DataflowSchema)
    return (
        dataflow_id = schema.dataflow_info.id,
        agency = schema.dataflow_info.agency,
        required_columns = get_required_columns(schema),
        available_dimensions = get_dimension_order(schema)
    )
end

"""
    prepare_source_structure(mappings::SDMXMappingResult) -> NamedTuple

Prepare source structure information from mapping results.
"""
function prepare_source_structure(mappings::SDMXMappingResult)
    return (
        columns = mappings.source_columns,
        mappings = mappings.suggested_mappings
    )
end

# Sample unique values from a column (internal function)
function _sample_unique(col::AbstractVector, n::Int=5)
    unique_vals = unique(col)
    return unique_vals[1:min(n, length(unique_vals))]
end

# Sample sheets for Excel analysis (internal function)
function _sample_sheets(file_path::String, sheet_names::Vector{String}, max_sheets::Int=3)
    sampled_sheets = sheet_names[1:min(max_sheets, length(sheet_names))]
    return NamedTuple(
        Symbol(sheet_name) => _read_sheet_sample(file_path, sheet_name)
        for sheet_name in sampled_sheets
    )
end

# Read sample data from a sheet (internal function)
function _read_sheet_sample(file_path::String, sheet_name::String)
    try
        data = XLSX.readtable(file_path, sheet_name, first_row=1, stop_in_empty_row=false)
        df = DataFrame(data)
        return (
            dimensions = size(df),
            columns = names(df)[1:min(10, ncol(df))],
            sample_data = first(df, min(5, nrow(df)))
        )
    catch e
        return (error = "Could not read sheet: $e",)
    end
end

# =================== PROMPT TEMPLATES ===================

"""
    create_mapping_template() -> PromptTemplate

Creates a reusable prompt template for column mapping tasks.
"""
function create_mapping_template()
    return PromptingTools.create_template(
        """You are an expert in SDMX (Statistical Data and Metadata eXchange) standards. 
        You analyze source data columns and suggest mappings to SDMX dimensions and attributes.
        
        Provide structured, confident mappings with clear reasoning.
        Focus on standard SDMX dimensions like FREQ, GEO_PICT, INDICATOR, TIME_PERIOD, OBS_VALUE.
        """,
        """Analyze the following source data and suggest SDMX column mappings:

        **Source Data Preview:**
        {{data_preview}}
        
        **Available SDMX Dimensions:**
        {{sdmx_dimensions}}
        
        **SDMX Schema Context:**
        {{schema_context}}
        
        For each source column, provide:
        1. Best SDMX dimension/attribute match
        2. Confidence score (0.0-1.0) 
        3. Required transformations
        4. Any concerns or notes
        
        Return analysis as SDMXMappingResult struct.""",
        load_as="sdmx_mapping"
    )
end

"""
    create_script_template() -> PromptTemplate

Creates a reusable prompt template for transformation script generation.
"""
function create_script_template()
    return PromptingTools.create_template(
        """You are an expert Julia programmer specializing in data transformation for SDMX compliance.
        Generate efficient, readable transformation scripts using DataFrames.jl or Tidier.jl.
        
        Follow Julia best practices:
        - Use descriptive variable names
        - Include helpful comments
        - Handle missing values appropriately
        - Use efficient DataFrame operations
        """,
        """Generate a Julia transformation script for the following SDMX mapping task:

        **Source Data Structure:**
        {{source_structure}}
        
        **Target SDMX Schema:**
        {{target_schema}}
        
        **Column Mappings:**
        {{column_mappings}}
        
        **Transformation Requirements:**
        {{transformation_requirements}}
        
        **Preferred Style:**
        {{script_style}}  # "dataframes" or "tidier"
        
        Generate a complete, executable Julia script that:
        1. Reads the source data
        2. Applies all necessary transformations
        3. Validates the output against SDMX requirements
        4. Exports clean SDMX-compliant data
        
        Return as SDMXTransformationScript struct.""",
        load_as="sdmx_script"
    )
end

"""
    create_excel_analysis_template() -> PromptTemplate

Creates a reusable prompt template for Excel structure analysis.
"""
function create_excel_analysis_template()
    return PromptingTools.create_template(
        """You are an expert at analyzing complex Excel files for data extraction.
        You understand various Excel structures including:
        - Multi-sheet workbooks with metadata
        - Pivot tables and cross-tabulated data
        - Headers spanning multiple rows
        - Mixed data types and formatting
        """,
        """Analyze this Excel file structure for optimal data extraction:

        **File Information:**
        {{file_info}}
        
        **Sheet Overview:**
        {{sheet_info}}
        
        **Sample Data from Each Sheet:**
        {{sample_data}}
        
        Determine:
        1. Which sheet contains the main data
        2. Exact data range (start/end rows and columns)
        3. Header structure and location
        4. Whether data is in pivot/cross-tab format
        5. Any data quality issues
        6. Recommended preprocessing steps
        
        Return as ExcelStructureAnalysis struct.""",
        load_as="excel_analysis"
    )
end

# =================== UNIFIED SDMX LLM INTERFACE ===================

"""
    sdmx_aigenerate(prompt::String; provider::Symbol=:ollama, model::String="", kwargs...) -> AIMessage

Generate text using specified LLM provider with SDMX-optimized settings.

This function provides a unified interface for generating text using various LLM providers,
with optimized settings for SDMX data transformation tasks. It automatically configures
provider-specific parameters and handles authentication when needed.

# Arguments
- `prompt::String`: The input prompt for text generation
- `provider::Symbol=:ollama`: LLM provider (:ollama, :openai, :anthropic, :google, etc.)
- `model::String=""`: Specific model name (uses provider defaults if empty)
- `kwargs...`: Additional provider-specific parameters

# Returns
- `AIMessage`: Generated text response from the LLM

# Examples
```julia
# Basic usage with Ollama (local)
response = sdmx_aigenerate("Explain SDMX data structure")

# Use OpenAI with specific model
response = sdmx_aigenerate(
    "Generate mapping suggestions for this data",
    provider=:openai,
    model="gpt-4"
)

# Use with custom parameters
response = sdmx_aigenerate(
    prompt,
    provider=:anthropic,
    temperature=0.1,
    max_tokens=1000
)
```

# Throws
- `ArgumentError`: If provider is not supported
- `HTTP.ExceptionRequest.StatusError`: If API authentication fails

# See also
[`sdmx_aiextract`](@ref), [`setup_sdmx_llm`](@ref), [`generate_transformation_script`](@ref)
"""
function sdmx_aigenerate(prompt::String; provider::Symbol=:ollama, model::String="", kwargs...)
    @assert !isempty(prompt) "Prompt cannot be empty"
    
    provider_info = get(SDMX_PROVIDERS, provider, nothing)
    @assert provider_info !== nothing "Unsupported provider: " * string(provider) * ". Available: " * join(keys(SDMX_PROVIDERS), ", ")
    @assert :generation in provider_info.supports "Provider " * string(provider) * " doesn't support text generation"
    
    # Use default model if none specified
    model_name = isempty(model) ? provider_info.default_model : model
    
    return PromptingTools.aigenerate(provider_info.schema, prompt; model=model_name, kwargs...)
end

"""
    sdmx_aiextract(return_type::Type, prompt::String; provider::Symbol=:openai, model::String="", kwargs...)

Extract structured data using specified provider. Falls back to text generation for unsupported providers.
"""
function sdmx_aiextract(return_type::Type, prompt::String; provider::Symbol=:openai, model::String="", kwargs...)
    @assert !isempty(prompt) "Prompt cannot be empty"
    
    provider_info = get(SDMX_PROVIDERS, provider, nothing)
    @assert provider_info !== nothing "Unsupported provider: " * string(provider)
    
    model_name = isempty(model) ? provider_info.default_model : model
    
    if :extraction in provider_info.supports
        return PromptingTools.aiextract(provider_info.schema, prompt; model=model_name, return_type=return_type, kwargs...)
    else
        @warn "Provider " * string(provider) * " doesn't support extraction, falling back to generation"
        # Format prompt for manual extraction
        structured_prompt = prompt * """
        
        Please format your response as valid Julia data that can be parsed into the requested type: """ * string(return_type)
        
        result = PromptingTools.aigenerate(provider_info.schema, structured_prompt; model=model_name, kwargs...)
        return result.content  # Return raw content for manual parsing
    end
end

# =================== AI-POWERED ANALYSIS FUNCTIONS ===================

"""
    analyze_excel_with_ai(file_path::String; model::String="sdmx_model") -> ExcelStructureAnalysis

Uses AI to analyze Excel file structure and recommend optimal extraction strategy.
"""
function analyze_excel_with_ai(file_path::String; model::String="sdmx_model")
    # Read Excel metadata functionally
    xf = XLSX.readxlsx(file_path)
    sheet_names = XLSX.sheetnames(xf)
    
    # Prepare file information as named tuple
    file_info = (
        filename = basename(file_path),
        num_sheets = length(sheet_names),
        sheet_names = sheet_names
    )
    
    # Sample sheets functionally
    sheet_samples = _sample_sheets(file_path, sheet_names)
    
    # Use AI to analyze structure
    template = create_excel_analysis_template()
    
    return aiextract(
        ExcelStructureAnalysis,
        template;
        file_info = file_info,
        sheet_info = sheet_names,
        sample_data = sheet_samples,
        model = model
    )
end

"""
    infer_column_mappings(data::DataFrame, schema::DataflowSchema; 
                         model::String="sdmx_model") -> SDMXMappingResult

Uses AI to suggest optimal column mappings from source data to SDMX schema.
"""
function infer_column_mappings(data::DataFrame, schema::DataflowSchema; model::String="sdmx_model")
    # Prepare data and schema contexts functionally
    data_preview = prepare_data_preview(data)
    schema_context = prepare_schema_context(schema)
    
    # Extract dimensions for context
    sdmx_dimensions = schema_context.available_dimensions
    
    # Use AI to infer mappings
    template = create_mapping_template()
    
    return aiextract(
        SDMXMappingResult,
        template;
        data_preview = data_preview,
        sdmx_dimensions = sdmx_dimensions,
        schema_context = schema_context,
        model = model
    )
end

"""
    generate_transformation_script(mappings::SDMXMappingResult, schema::DataflowSchema;
                                  script_style::Union{Symbol, ScriptStyle}=:dataframes, 
                                  model::String="sdmx_model") -> SDMXTransformationScript

Uses AI to generate a complete Julia transformation script based on column mappings.
"""
function generate_transformation_script(mappings::SDMXMappingResult, schema::DataflowSchema;
                                       script_style::Union{Symbol, ScriptStyle}=:dataframes,
                                       model::String="sdmx_model")
    
    # Convert symbol to enum if needed
    style_enum = isa(script_style, Symbol) ? script_style_enum(script_style) : script_style
    
    # Prepare contexts functionally
    source_structure = prepare_source_structure(mappings)
    target_schema = prepare_schema_context(schema)
    
    # Default transformation requirements
    transformation_requirements = (
        "Handle missing values appropriately",
        "Validate against SDMX requirements", 
        "Include data quality checks",
        "Export to SDMX-CSV format"
    )
    
    # Use AI to generate script
    template = create_script_template()
    
    return aiextract(
        SDMXTransformationScript,
        template;
        source_structure = source_structure,
        target_schema = target_schema,
        column_mappings = mappings.suggested_mappings,
        transformation_requirements = transformation_requirements,
        script_style = style_enum,
        model = model
    )
end

# =================== CONVENIENCE FUNCTIONS ===================

"""
    ai_sdmx_workflow(file_path::String, schema::DataflowSchema; 
                    model::String="sdmx_model",
                    script_style::Union{Symbol, ScriptStyle}=:dataframes) -> NamedTuple

Complete AI-powered workflow: Excel analysis → column mapping → script generation.
Returns named tuple with all workflow results.
"""
function ai_sdmx_workflow(file_path::String, schema::DataflowSchema; 
                         model::String="sdmx_model",
                         script_style::Union{Symbol, ScriptStyle}=:dataframes)
    
    @info "Starting AI-powered SDMX workflow for $file_path"
    
    # Execute workflow steps functionally
    excel_analysis = _analyze_excel_step(file_path, model)
    data = _load_data_step(file_path, excel_analysis)
    mappings = _infer_mappings_step(data, schema, model)
    script = _generate_script_step(mappings, schema, script_style, model)
    
    @info "AI workflow completed successfully!"
    
    return (
        excel_analysis = excel_analysis,
        data = data,
        mappings = mappings,
        script = script
    )
end

# Execute Excel analysis step (internal function)
function _analyze_excel_step(file_path::String, model::String)
    @info "Step 1: Analyzing Excel structure..."
    return analyze_excel_with_ai(file_path; model=model)
end

# Execute data loading step (internal function)
function _load_data_step(file_path::String, excel_analysis)
    @info "Step 2: Loading data from recommended sheet: \$(excel_analysis.recommended_sheet)"
    return XLSX.readtable(file_path, excel_analysis.recommended_sheet, 
                         first_row=excel_analysis.header_row) |> DataFrame
end

# Execute mapping inference step (internal function)
function _infer_mappings_step(data::DataFrame, schema, model::String)
    @info "Step 3: Inferring column mappings..."
    return infer_column_mappings(data, schema; model=model)
end

# Execute script generation step (internal function)
function _generate_script_step(mappings, schema, script_style, model::String)
    @info "Step 4: Generating transformation script..."
    return generate_transformation_script(mappings, schema; 
                                        script_style=script_style, model=model)
end

# =================== STRING-BASED ALTERNATIVE FUNCTIONS ===================
# These provide simpler text-based interfaces as alternatives to structured responses

"""
    infer_sdmx_column_mappings(source_columns::Vector{String}, target_schema; 
                              provider::Symbol=:ollama, model::String="")

Infer optimal column mappings from source data to SDMX schema using LLM analysis.
Returns text-based mapping suggestions rather than structured data.

This is a simpler alternative to `infer_column_mappings()` that returns plain text.

# Arguments
- `source_columns`: Vector of source column names
- `target_schema`: Either a DataflowSchema or Dict with :dimensions and :measures
- `provider`: LLM provider to use
- `model`: Specific model (uses provider default if empty)

# Returns
- `AIMessage`: LLM response with mapping suggestions as text

# Examples
```julia
columns = ["DATE", "COUNTRY", "GDP_VALUE"]
schema = Dict(:dimensions => ["TIME_PERIOD", "GEO_PICT"], :measures => ["OBS_VALUE"])
response = infer_sdmx_column_mappings(columns, schema)
println(response.content)
```
"""
function infer_sdmx_column_mappings(source_columns::Vector{String}, target_schema; 
                                   provider::Symbol=:ollama, model::String="")
    @assert !isempty(source_columns) "Source columns cannot be empty"
    
    # Extract schema information - handle both DataflowSchema and Dict
    if hasfield(typeof(target_schema), :dimensions) && hasfield(typeof(target_schema), :measures)
        # Handle DataflowSchema
        dimensions = target_schema.dimensions.dimension_id
        measures = target_schema.measures.measure_id
    elseif isa(target_schema, Dict)
        # Handle Dict format
        dimensions = target_schema[:dimensions]
        measures = get(target_schema, :measures, ["OBS_VALUE"])
    else
        error("Invalid target schema type: " * string(typeof(target_schema)))
    end
    
    prompt = """
    You are an SDMX (Statistical Data and Metadata eXchange) expert. 
    
    Source data columns: """ * join(source_columns, ", ") * """
    
    Target SDMX dimensions: """ * join(dimensions, ", ") * """
    Target SDMX measures: """ * join(measures, ", ") * """
    
    Suggest optimal column mappings from source to SDMX schema.
    Focus on standard SDMX patterns:
    - Geographic areas → GEO_PICT 
    - Time periods → TIME_PERIOD
    - Statistical indicators → INDICATOR
    - Observed values → OBS_VALUE
    - Units of measure → UNIT_MEASURE
    
    For each mapping, provide:
    1. Source column → Target SDMX dimension/measure
    2. Confidence level (high/medium/low)
    3. Required transformations or notes
    
    Be specific and actionable.
    """
    
    # Get provider configuration
    provider_enum = isa(provider, Symbol) ? llm_provider(provider) : provider
    model_name = isempty(model) ? _get_default_model(provider_enum, model) : model
    
    # Use PromptingTools directly for simpler text response
    return PromptingTools.aigenerate(prompt; model=model_name)
end

"""
    generate_transformation_script_text(mappings::String, schema_info; 
                                       provider::Symbol=:ollama, model::String="", 
                                       style::Symbol=:tidier, 
                                       excel_analysis::Union{ExcelStructureAnalysis, Nothing}=nothing)

Generate Julia transformation script from text-based mappings.
This is a simpler alternative to the structured version that accepts mapping text.

# Arguments
- `mappings`: Text description of column mappings
- `schema_info`: Target SDMX schema (DataflowSchema or string description)
- `provider`: LLM provider
- `model`: Specific model
- `style`: Script style (:tidier, :dataframes, :mixed)
- `excel_analysis`: Optional Excel analysis results

# Returns
- `AIMessage`: Generated script as text

# Examples
```julia
mappings = "Map DATE to TIME_PERIOD, COUNTRY to GEO_PICT..."
script = generate_transformation_script_text(mappings, schema)
println(script.content)
```
"""
function generate_transformation_script_text(mappings::String, schema_info; 
                                            provider::Symbol=:ollama, model::String="", 
                                            style::Symbol=:tidier, 
                                            excel_analysis::Union{ExcelStructureAnalysis, Nothing}=nothing)
    @assert !isempty(mappings) "Column mappings cannot be empty"
    @assert style in [:tidier, :dataframes, :mixed] "Style must be :tidier, :dataframes, or :mixed"
    
    style_instruction = if style == :tidier
        "Use Tidier.jl syntax (@select, @mutate, @filter, @pivot_longer, etc.)"
    elseif style == :dataframes
        "Use DataFrames.jl syntax (select, transform, filter, etc.)"
    else
        "Use a mix of DataFrames.jl and Tidier.jl as appropriate"
    end
    
    # Add Excel analysis context if available
    excel_context = if excel_analysis !== nothing
        """
        
        Excel Analysis Context:
        - Complexity Score: """ * string(excel_analysis.complexity_score) * """
        - Pivoting Detected: """ * string(excel_analysis.pivoting_detected) * """
        - Recommended Sheet: """ * excel_analysis.recommended_sheet * """
        - Transformation Hints: """ * join(excel_analysis.transformation_hints, "; ")
    else
        ""
    end
    
    schema_description = if isa(schema_info, DataflowSchema)
        "Dataflow: " * schema_info.dataflow_info.name
    else
        string(schema_info)
    end
    
    prompt = """
    Generate a complete Julia script to transform data for SDMX compliance.
    
    Column mappings identified:
    """ * mappings * """
    
    Target SDMX schema information:
    """ * schema_description * excel_context * """
    
    Requirements:
    1. """ * style_instruction * """
    2. Read source data (assume CSV format unless Excel context suggests otherwise)
    3. Apply all necessary transformations based on the mappings
    4. Handle missing data appropriately with @assert statements
    5. Validate output against SDMX requirements  
    6. Export clean SDMX-CSV format
    7. Include comprehensive error handling
    8. Add clear comments explaining each transformation step
    
    Generate complete, executable Julia code only.
    """
    
    # Get provider configuration
    provider_enum = isa(provider, Symbol) ? llm_provider(provider) : provider
    model_name = isempty(model) ? _get_default_model(provider_enum, model) : model
    
    # Use PromptingTools directly for simpler text response
    return PromptingTools.aigenerate(prompt; model=model_name)
end