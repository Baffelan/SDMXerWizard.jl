"""
Comprehensive SDMX Metadata Context Builder

This module extracts rich structural information from SDMX schemas and data sources
without sending actual data to LLMs. It leverages all available SDMXer.jl functions
to build comprehensive context for intelligent transformation script generation.
"""

# Dependencies loaded at package level


# =================== CONTEXT STRUCTURES ===================

"""
Comprehensive SDMX structural information extracted from schemas and availability
"""
struct SDMXStructuralContext
    # Basic schema info
    dataflow_info::NamedTuple
    dimensions::DataFrame
    measures::DataFrame
    attributes::DataFrame
    time_dimension::Union{NamedTuple, Nothing}
    
    # Rich codelist information
    codelists::DataFrame  # All codes with hierarchies
    codelist_summary::Dict{String, NamedTuple}  # Stats per codelist
    available_codes::Dict{String, Vector{String}}  # Actually available codes
    
    # Conceptual information
    concepts::DataFrame  # Concept definitions and roles
    concept_descriptions::Dict{String, String}
    
    # Availability constraints
    availability::Union{AvailabilityConstraint, Nothing}
    data_coverage::Dict{String, Any}
    time_coverage::Union{TimeAvailability, Nothing}
    
    # Validation requirements
    required_columns::Vector{String}
    optional_columns::Vector{String}
    codelist_columns::Dict{String, String}  # column -> codelist_id
    dimension_order::Vector{String}
end

"""
Data source structural information (no actual data sent to LLM)
"""
struct DataSourceContext
    # File information
    file_info::NamedTuple
    source_profile::SourceDataProfile
    
    # Column analysis (structure only)
    column_patterns::Dict{String, NamedTuple}  # Pattern analysis per column
    data_shape::NamedTuple  # Dimensions, pivoting needs, etc.
    
    # Excel-specific analysis
    excel_structure::Union{ExcelStructureAnalysis, Nothing}
    sheet_analysis::Dict{String, NamedTuple}
    
    # Detected patterns
    time_patterns::Vector{NamedTuple}  # Time formats found
    geographic_patterns::Vector{String}  # Country/region codes found
    value_patterns::Dict{String, NamedTuple}  # Numeric patterns, units
    hierarchical_patterns::Vector{NamedTuple}  # Parent-child relationships
end

"""
Combined context for transformation script generation
"""
struct TransformationContext
    sdmx_context::SDMXStructuralContext
    source_context::DataSourceContext
    mapping_suggestions::DataFrame  # Automated mapping suggestions
    transformation_requirements::Vector{NamedTuple}  # Required transformations
    complexity_assessment::NamedTuple  # Complexity scoring
end

# =================== SDMX STRUCTURAL CONTEXT EXTRACTION ===================

"""
    extract_structural_context(dataflow_url::String) -> SDMXStructuralContext

Extract comprehensive structural information from SDMX dataflow schema.
"""
function extract_structural_context(dataflow_url::String)
    @assert !isempty(dataflow_url) "Dataflow URL cannot be empty"
    @assert is_url(dataflow_url) "Input must be a valid URL"
    
    # Extract basic schema using SDMXer.jl functions
    schema = extract_dataflow_schema(dataflow_url)
    @assert schema !== nothing "Failed to extract dataflow schema from $dataflow_url"
    @assert !isempty(schema.dataflow_info.id) "Dataflow must have a valid ID"
    
    # Extract all codelists with full hierarchy
    codelists = extract_all_codelists(dataflow_url)
    @assert nrow(codelists) > 0 "No codelists found in dataflow schema"
    
    # Extract concept definitions
    concepts = extract_concepts(dataflow_url)
    @assert nrow(concepts) > 0 "No concepts found in dataflow schema"
    
    # Build codelist summary with statistics
    codelist_summary = build_codelist_summary(codelists)
    @assert !isempty(codelist_summary) "Failed to build codelist summary"
    
    # Extract availability constraints if available
    availability_url = construct_availability_url(dataflow_url)
    availability = nothing
    available_codes = Dict{String, Vector{String}}()
    data_coverage = Dict{String, Any}()
    time_coverage = nothing
    
    if !isempty(availability_url)
        availability = extract_availability(availability_url)
        if availability !== nothing
            available_codes = extract_available_codes_by_dimension(availability)
            data_coverage = get_data_coverage_summary(availability)
            time_coverage = get_time_coverage(availability)
            @assert !isempty(available_codes) "Availability extracted but no available codes found"
        end
    end
    
    # Build concept descriptions dictionary
    concept_descriptions = build_concept_descriptions(concepts)
    @assert !isempty(concept_descriptions) "No concept descriptions found"
    
    # Extract column requirements
    required_columns = get_required_columns(schema)
    optional_columns = get_optional_columns(schema)
    codelist_columns = get_codelist_columns(schema)
    dimension_order = get_dimension_order(schema)
    
    @assert !isempty(required_columns) "Schema must have at least one required column"
    @assert !isempty(dimension_order) "Schema must specify dimension order"
    
    return SDMXStructuralContext(
        schema.dataflow_info,
        schema.dimensions,
        schema.measures,
        schema.attributes,
        schema.time_dimension,
        codelists,
        codelist_summary,
        available_codes,
        concepts,
        concept_descriptions,
        availability,
        data_coverage,
        time_coverage,
        required_columns,
        optional_columns,
        codelist_columns,
        dimension_order
    )
end

# =================== DATA SOURCE CONTEXT EXTRACTION ===================

"""
    extract_data_source_context(file_path::String) -> DataSourceContext

Extract comprehensive data source structure without sending actual data to LLM.
"""
function extract_data_source_context(file_path::String)
    @assert !isempty(file_path) "File path cannot be empty"
    @assert isfile(file_path) "File does not exist: $file_path"
    
    file_size = stat(file_path).size
    @assert file_size > 0 "File cannot be empty: $file_path"
    @assert file_size < 1_000_000_000 "File too large (>1GB): $file_path"  # Safety check
    
    # Read and profile the source data using SDMXer.jl functions
    source_data = read_source_data(file_path)
    @assert nrow(source_data) > 0 "Source data cannot be empty"
    @assert ncol(source_data) > 0 "Source data must have columns"
    
    source_profile = profile_source_data(source_data)
    @assert !isempty(source_profile.columns) "Source profile must contain column information"
    @assert source_profile.row_count > 0 "Source profile must show positive row count"
    
    # File information
    file_info = (
        path = file_path,
        size_mb = round(file_size / 1024 / 1024, digits=2),
        extension = splitext(file_path)[2],
        modified = Dates.unix2datetime(stat(file_path).mtime)
    )
    
    # Analyze column patterns (structure only, no actual values)
    column_patterns = analyze_column_patterns(source_profile)
    @assert !isempty(column_patterns) "Failed to analyze column patterns"
    
    # Analyze data shape and structure
    data_shape = analyze_data_shape(source_data, source_profile)
    @assert haskey(data_shape, :complexity_score) "Data shape analysis must include complexity score"
    
    # Excel-specific analysis if applicable
    excel_structure = nothing
    sheet_analysis = Dict{String, NamedTuple}()
    
    if endswith(lowercase(file_path), ".xlsx") || endswith(lowercase(file_path), ".xls")
        excel_structure = analyze_excel_structure(file_path)
        sheet_analysis = analyze_excel_sheets(file_path)
        @assert !isempty(sheet_analysis) "Excel file should have analyzable sheets"
    end
    
    # Pattern detection (using statistical analysis, not actual values)
    time_patterns = detect_time_patterns(source_profile)
    geographic_patterns = detect_geographic_patterns(source_profile)
    value_patterns = detect_value_patterns(source_profile)
    hierarchical_patterns = detect_hierarchical_patterns(source_data, source_profile)
    
    return DataSourceContext(
        file_info,
        source_profile,
        column_patterns,
        data_shape,
        excel_structure,
        sheet_analysis,
        time_patterns,
        geographic_patterns,
        value_patterns,
        hierarchical_patterns
    )
end

# =================== CONTEXT BUILDING HELPERS ===================

"""
Build summary statistics for each codelist
"""
function build_codelist_summary(codelists::DataFrame)
    @assert nrow(codelists) > 0 "Codelists DataFrame cannot be empty"
    @assert "codelist_id" in names(codelists) "Codelists must have codelist_id column"
    @assert "code_id" in names(codelists) "Codelists must have code_id column"
    
    summary = Dict{String, NamedTuple}()
    unique_codelists = unique(codelists.codelist_id)
    @assert !isempty(unique_codelists) "Must have at least one unique codelist"
    
    for codelist_id in unique_codelists
        codes = filter(row -> row.codelist_id == codelist_id, codelists)
        @assert nrow(codes) > 0 "Each codelist must have at least one code"
        
        # Analyze hierarchy depth
        has_hierarchy = "parent_code_id" in names(codes) && any(!ismissing(codes.parent_code_id))
        max_depth = has_hierarchy ? calculate_hierarchy_depth(codes) : 1
        
        # Language coverage
        languages = "lang" in names(codes) ? unique(filter(!ismissing, codes.lang)) : String[]
        
        sample_codes = first(codes.code_id, min(5, nrow(codes)))
        @assert !isempty(sample_codes) "Must have sample codes"
        
        summary[codelist_id] = (
            total_codes = nrow(codes),
            has_hierarchy = has_hierarchy,
            max_depth = max_depth,
            languages = languages,
            sample_codes = sample_codes
        )
    end
    
    return summary
end

"""
Extract available codes by dimension from availability constraints
"""
function extract_available_codes_by_dimension(availability::AvailabilityConstraint)
    @assert !isempty(availability.dimensions) "Availability constraint must have dimensions"
    
    available = Dict{String, Vector{String}}()
    
    for dim in availability.dimensions
        @assert !isempty(dim.dimension_id) "Dimension must have valid ID"
        @assert !isempty(dim.available_values) "Dimension must have available values"
        available[dim.dimension_id] = dim.available_values
    end
    
    return available
end

"""
Build concept descriptions dictionary
"""
function build_concept_descriptions(concepts::DataFrame)
    @assert nrow(concepts) > 0 "Concepts DataFrame cannot be empty"
    @assert "concept_id" in names(concepts) "Concepts must have concept_id column"
    
    descriptions = Dict{String, String}()
    
    for row in eachrow(concepts)
        @assert !isempty(row.concept_id) "Concept ID cannot be empty"
        
        if "description" in names(concepts) && !ismissing(row.description) && !isempty(row.description)
            descriptions[row.concept_id] = row.description
        end
    end
    
    return descriptions
end

"""
Analyze column patterns without exposing actual data
"""
function analyze_column_patterns(profile::SourceDataProfile)
    @assert !isempty(profile.columns) "Profile must have column information"
    @assert profile.row_count > 0 "Profile must have positive row count"
    
    patterns = Dict{String, NamedTuple}()
    
    for col in profile.columns
        @assert !isempty(col.name) "Column must have a name"
        @assert col.missing_count >= 0 "Missing count cannot be negative"
        @assert col.unique_count > 0 "Unique count must be positive"
        
        missing_ratio = col.missing_count / profile.row_count
        unique_ratio = col.unique_count / profile.row_count
        
        @assert 0 <= missing_ratio <= 1 "Missing ratio must be between 0 and 1"
        @assert 0 < unique_ratio <= 1 "Unique ratio must be between 0 and 1"
        
        patterns[col.name] = (
            data_type = col.type,
            missing_ratio = missing_ratio,
            is_categorical = col.is_categorical,
            is_temporal = col.is_temporal,
            is_numeric = col.numeric_stats !== nothing,
            unique_ratio = unique_ratio,
            categorical_info = col.is_categorical ? (
                categories = col.unique_count,
                is_ordered = detect_ordering_pattern(col)
            ) : nothing,
            temporal_info = col.is_temporal ? (
                format = col.temporal_format,
                frequency = detect_temporal_frequency(col)
            ) : nothing,
            numeric_info = col.numeric_stats !== nothing ? (
                range = (col.numeric_stats.min, col.numeric_stats.max),
                distribution = analyze_numeric_distribution(col.numeric_stats)
            ) : nothing
        )
    end
    
    return patterns
end

"""
Analyze data shape and pivoting requirements
"""
function analyze_data_shape(data::DataFrame, profile::SourceDataProfile)
    @assert nrow(data) > 0 "Data cannot be empty"
    @assert ncol(data) > 0 "Data must have columns"
    @assert profile.row_count == nrow(data) "Profile row count must match data"
    
    # Detect if data is in wide or long format
    is_wide_format = detect_wide_format(data, profile)
    
    # Detect potential pivoting columns
    pivot_candidates = detect_pivot_candidates(data, profile)
    
    # Detect header rows in data
    header_issues = detect_header_issues(data)
    @assert all(issue >= 1 for issue in header_issues) "Header issue row numbers must be positive"
    
    estimated_data_rows = profile.row_count - length(header_issues)
    @assert estimated_data_rows > 0 "Must have some data rows after removing headers"
    
    complexity_score = calculate_shape_complexity(is_wide_format, pivot_candidates, header_issues)
    @assert complexity_score >= 0 "Complexity score cannot be negative"
    
    return (
        is_wide_format = is_wide_format,
        needs_pivoting = !isempty(pivot_candidates),
        pivot_candidates = pivot_candidates,
        header_issues = header_issues,
        estimated_data_rows = estimated_data_rows,
        complexity_score = complexity_score
    )
end

# =================== PATTERN DETECTION FUNCTIONS ===================

function detect_time_patterns(profile::SourceDataProfile)
    patterns = NamedTuple[]
    
    for col in profile.columns
        if col.is_temporal
            @assert !isempty(col.temporal_format) "Temporal column must have format information"
            
            pattern = (
                column = col.name,
                format = col.temporal_format,
                frequency = detect_temporal_frequency(col),
                coverage = estimate_temporal_coverage(col)
            )
            push!(patterns, pattern)
        end
    end
    
    return patterns
end

function detect_geographic_patterns(profile::SourceDataProfile)
    patterns = String[]
    
    for col in profile.columns
        if is_geographic_column(col)
            push!(patterns, col.name)
        end
    end
    
    return patterns
end

function detect_value_patterns(profile::SourceDataProfile)
    patterns = Dict{String, NamedTuple}()
    
    for col in profile.columns
        if col.numeric_stats !== nothing
            @assert col.numeric_stats.min <= col.numeric_stats.max "Min must be <= max"
            
            max_abs = max(abs(col.numeric_stats.min), abs(col.numeric_stats.max))
            @assert max_abs >= 0 "Maximum absolute value cannot be negative"
            
            magnitude = max_abs > 0 ? log10(max_abs) : 0
            
            patterns[col.name] = (
                scale = detect_numeric_scale(col.numeric_stats),
                likely_unit = detect_likely_unit(col),
                has_negatives = col.numeric_stats.min < 0,
                has_decimals = detect_decimal_usage(col.numeric_stats),
                magnitude = magnitude
            )
        end
    end
    
    return patterns
end

function detect_hierarchical_patterns(data::DataFrame, profile::SourceDataProfile)
    patterns = NamedTuple[]
    
    # Look for parent-child relationships in the data structure
    categorical_cols = [col.name for col in profile.columns if col.is_categorical]
    
    for i in 1:length(categorical_cols)
        for j in (i+1):length(categorical_cols)
            col1, col2 = categorical_cols[i], categorical_cols[j]
            @assert col1 in names(data) "Column $col1 must exist in data"
            @assert col2 in names(data) "Column $col2 must exist in data"
            
            if detect_hierarchy_relationship(data, col1, col2)
                strength = calculate_hierarchy_strength(data, col1, col2)
                @assert 0 <= strength <= 1 "Hierarchy strength must be between 0 and 1"
                
                pattern = (
                    parent_column = col1,
                    child_column = col2,
                    relationship_strength = strength
                )
                push!(patterns, pattern)
            end
        end
    end
    
    return patterns
end

# =================== HELPER FUNCTIONS ===================

function detect_ordering_pattern(col::ColumnProfile)
    @assert col.is_categorical "Column must be categorical"
    return col.unique_count < 20  # Simple heuristic
end

function detect_temporal_frequency(col::ColumnProfile)
    return "unknown"  # Placeholder - would implement frequency detection
end

function analyze_numeric_distribution(stats)
    @assert stats.min <= stats.max "Invalid numeric stats"
    
    range = stats.max - stats.min
    @assert range >= 0 "Range cannot be negative"
    
    return (
        is_positive_only = stats.min >= 0,
        range_magnitude = range > 0 ? log10(range + 1) : 0,
        likely_percentage = stats.min >= 0 && stats.max <= 100
    )
end

function detect_wide_format(data::DataFrame, profile::SourceDataProfile)
    # Detect if data is in wide format (many columns, fewer rows)
    return ncol(data) > 10 && nrow(data) < ncol(data) * 5
end

function detect_pivot_candidates(data::DataFrame, profile::SourceDataProfile)
    candidates = String[]
    
    for col in profile.columns
        if col.is_temporal || (col.is_categorical && col.unique_count < 50)
            push!(candidates, col.name)
        end
    end
    
    return candidates
end

function detect_header_issues(data::DataFrame)
    # Detect non-data rows at the top
    issues = Int[]
    # Placeholder - would implement header detection logic
    return issues
end

function calculate_shape_complexity(is_wide::Bool, pivot_candidates::Vector{String}, header_issues::Vector{Int})
    score = 0.0
    score += is_wide ? 2.0 : 0.0
    score += length(pivot_candidates) * 0.5
    score += length(header_issues) * 1.0
    return score
end

function is_geographic_column(col::ColumnProfile)
    @assert !isempty(col.name) "Column name cannot be empty"
    
    name_lower = lowercase(col.name)
    return any(keyword -> occursin(keyword, name_lower), 
               ["country", "region", "geo", "area", "location", "place"])
end

function detect_numeric_scale(stats)
    max_val = max(abs(stats.min), abs(stats.max))
    
    if max_val < 1
        return "fractional"
    elseif max_val < 1000
        return "units"
    elseif max_val < 1_000_000
        return "thousands"
    elseif max_val < 1_000_000_000
        return "millions"
    else
        return "billions_plus"
    end
end

function detect_likely_unit(col::ColumnProfile)
    @assert !isempty(col.name) "Column name cannot be empty"
    
    name_lower = lowercase(col.name)
    
    if occursin("gdp", name_lower) || occursin("income", name_lower)
        return "currency"
    elseif occursin("population", name_lower) || occursin("people", name_lower)
        return "persons"
    elseif occursin("percent", name_lower) || occursin("rate", name_lower)
        return "percentage"
    else
        return "unknown"
    end
end

function detect_decimal_usage(stats)
    # Heuristic: check if values appear to use decimals
    return (stats.max - stats.min) < 1000
end

function detect_hierarchy_relationship(data::DataFrame, col1::String, col2::String)
    # Placeholder for hierarchy detection
    return false
end

function calculate_hierarchy_strength(data::DataFrame, col1::String, col2::String)
    # Placeholder for hierarchy strength calculation
    return 0.0
end

function calculate_hierarchy_depth(codes::DataFrame)
    @assert nrow(codes) > 0 "Codes DataFrame cannot be empty"
    # Placeholder - would implement actual depth calculation
    return 1
end

function estimate_temporal_coverage(col::ColumnProfile)
    @assert col.is_temporal "Column must be temporal"
    @assert col.unique_count > 0 "Must have at least one unique temporal value"
    
    return (
        likely_start = "unknown",
        likely_end = "unknown", 
        estimated_periods = col.unique_count
    )
end

function analyze_excel_sheets(file_path::String)
    @assert endswith(lowercase(file_path), ".xlsx") || endswith(lowercase(file_path), ".xls") "Must be Excel file"
    @assert isfile(file_path) "Excel file must exist"
    
    analysis = Dict{String, NamedTuple}()
    
    xlsx = XLSX.readxlsx(file_path)
    sheet_names = XLSX.sheetnames(xlsx)
    @assert !isempty(sheet_names) "Excel file must have at least one sheet"
    
    for sheet_name in sheet_names
        @assert !isempty(sheet_name) "Sheet name cannot be empty"
        
        sheet = xlsx[sheet_name]
        dimensions = XLSX.get_dimension(sheet)
        
        analysis[sheet_name] = (
            dimensions = dimensions,
            has_data = !isempty(dimensions),
            estimated_header_rows = 1,  # Placeholder
            estimated_data_start = (2, 1)  # Placeholder
        )
    end
    
    return analysis
end