"""
Julia-idiomatic Data Source Type Hierarchy for SDMXer.jl

This module defines an abstract type hierarchy for different data sources,
leveraging Julia's multiple dispatch for clean, extensible data ingestion.
"""

# Dependencies loaded at package level

# =================== ABSTRACT TYPE HIERARCHY ===================

"""
    DataSource

Abstract base type for all data sources in SDMXer.jl.
All concrete data source types should inherit from this hierarchy.
"""
abstract type DataSource end

"""
    FileSource <: DataSource

Abstract type for file-based data sources (CSV, Excel, etc.).
"""
abstract type FileSource <: DataSource end

"""
    NetworkSource <: DataSource

Abstract type for network-based data sources (URLs, APIs, etc.).
"""
abstract type NetworkSource <: DataSource end

"""
    MemorySource <: DataSource

Abstract type for in-memory data sources (DataFrames, Arrays, etc.).
"""
abstract type MemorySource <: DataSource end

# =================== CONCRETE DATA SOURCE TYPES ===================

"""
    CSVSource <: FileSource

Represents a CSV file data source with configurable parsing options.

# Fields
- `path::String`: Path to the CSV file
- `delimiter::Char`: Column delimiter (default: ',')
- `encoding::String`: File encoding (default: "UTF-8")
- `header_row::Int`: Which row contains headers (default: 1)
- `skip_rows::Int`: Number of rows to skip (default: 0)
"""
struct CSVSource <: FileSource
    path::String
    delimiter::Char
    encoding::String
    header_row::Int
    skip_rows::Int
    
    # Inner constructor with sensible defaults
    function CSVSource(path::String; 
                      delimiter::Char = ',', 
                      encoding::String = "UTF-8",
                      header_row::Int = 1,
                      skip_rows::Int = 0)
        isfile(path) || throw(ArgumentError("File not found: " * path))
        endswith(lowercase(path), ".csv") || @warn "File does not have .csv extension: " * path
        new(path, delimiter, encoding, header_row, skip_rows)
    end
end

"""
    ExcelSource <: FileSource

Represents an Excel file data source with sheet and range selection.

# Fields
- `path::String`: Path to the Excel file
- `sheet::Union{String, Int}`: Sheet name or index (default: 1)
- `range::Union{String, Nothing}`: Cell range (e.g., "A1:D10", default: nothing for full sheet)
- `header_row::Int`: Which row contains headers (default: 1)
"""
struct ExcelSource <: FileSource
    path::String
    sheet::Union{String, Int}
    range::Union{String, Nothing}
    header_row::Int
    
    function ExcelSource(path::String; 
                        sheet::Union{String, Int}=1,
                        range::Union{String, Nothing}=nothing,
                        header_row::Int=1)
        isfile(path) || throw(ArgumentError("File not found: " * path))
        ext = lowercase(splitext(path)[2])
        ext in [".xlsx", ".xls"] || throw(ArgumentError("File must be Excel format (.xlsx/.xls): $path"))
        new(path, sheet, range, header_row)
    end
end

"""
    URLSource <: NetworkSource

Represents a network-based data source (HTTP/HTTPS URL).

# Fields
- `url::String`: The URL to fetch data from
- `headers::Dict{String, String}`: HTTP headers to include
- `timeout::Float64`: Request timeout in seconds
- `format::Symbol`: Expected data format (:csv, :excel, :xml, :json)
"""
struct URLSource <: NetworkSource
    url::String
    headers::Dict{String, String}
    timeout::Float64
    format::Symbol
    
    function URLSource(url::String; 
                      headers::Dict{String, String}=Dict{String, String}(),
                      timeout::Float64=30.0,
                      format::Symbol=:auto)
        startswith(url, "http") || throw(ArgumentError("URL must start with http:// or https://"))
        
        # Auto-detect format from URL if not specified
        if format == :auto
            if occursin(r"\\.csv($|\\?)", url)
                format = :csv
            elseif occursin(r"\\.(xlsx?|xls)($|\\?)", url)
                format = :excel
            elseif occursin(r"\\.xml($|\\?)", url)
                format = :xml
            elseif occursin(r"\\.json($|\\?)", url)
                format = :json
            else
                format = :unknown
            end
        end
        
        new(url, headers, timeout, format)
    end
end

"""
    DataFrameSource <: MemorySource

Represents an in-memory DataFrame as a data source.

# Fields
- `data::DataFrame`: The DataFrame containing the data
- `name::String`: A descriptive name for this data source
- `metadata::Dict{Symbol, Any}`: Additional metadata about the data
"""
struct DataFrameSource <: MemorySource
    data::DataFrame
    name::String
    metadata::Dict{Symbol, Any}
    
    function DataFrameSource(data::DataFrame, name::String="in_memory_data"; 
                           metadata::Dict{Symbol, Any}=Dict{Symbol, Any}())
        nrow(data) > 0 || @warn "DataFrame is empty"
        new(data, name, metadata)
    end
end

# =================== MULTIPLE DISPATCH INTERFACE ===================

"""
    read_data(source::DataSource) -> DataFrame

Reads data from any DataSource into a DataFrame using multiple dispatch.
Each concrete data source type implements its own reading logic.
"""
function read_data end

# CSV reading implementation
function read_data(source::CSVSource)
    try
        return CSV.read(source.path, DataFrame; 
                       delim=source.delimiter,
                       header=source.header_row,
                       skipto=source.header_row + source.skip_rows)
    catch e
        throw(ArgumentError("Failed to read CSV file $(source.path): $e"))
    end
end

# Excel reading implementation  
function read_data(source::ExcelSource)
    try
        if source.range !== nothing
            # Read specific range
            data = XLSX.readdata(source.path, source.sheet, source.range)
            # Convert to DataFrame (simplified - would need proper header handling)
            return DataFrame(data, :auto)
        else
            # Read entire sheet
            return XLSX.readtable(source.path, source.sheet; first_row=source.header_row) |> DataFrame
        end
    catch e
        throw(ArgumentError("Failed to read Excel file $(source.path): $e"))
    end
end

# URL reading implementation
function read_data(source::URLSource)
    try
        response = HTTP.get(source.url; headers=source.headers, timeout=source.timeout)
        
        if response.status != 200
            throw(ArgumentError("HTTP request failed with status $(response.status)"))
        end
        
        content = String(response.body)
        
        # Dispatch based on format
        if source.format == :csv
            return CSV.read(IOBuffer(content), DataFrame)
        elseif source.format == :xml
            # Return raw content for XML (to be parsed by SDMX functions)
            throw(ArgumentError("XML data from URLs should be processed by SDMX extraction functions, not read_data()"))
        else
            throw(ArgumentError("Unsupported format for URL data source: $(source.format)"))
        end
    catch e
        throw(ArgumentError("Failed to read data from URL $(source.url): $e"))
    end
end

# DataFrame reading implementation (identity function)
function read_data(source::DataFrameSource)
    return copy(source.data)  # Return a copy to avoid mutations
end

# =================== CONVENIENCE FUNCTIONS ===================

"""
    read_source_data(file_path::String; sheet = 1, header_row = 1) -> DataFrame

Convenience function to read data from CSV or Excel files.

This function provides a simple interface for reading data files,
automatically detecting the file type based on extension.

# Arguments
- `file_path::String`: Path to the data file
- `sheet = 1`: Excel sheet number or name (ignored for CSV)
- `header_row = 1`: Row number containing column headers

# Returns
- `DataFrame`: The loaded data

# Examples
```julia
# CSV files
df = read_source_data("data.csv")

# Excel files
df = read_source_data("data.xlsx"; sheet = "Sheet1", header_row = 2)
```
"""
function read_source_data(file_path::String; sheet = 1, header_row = 1)
    if !isfile(file_path)
        error("File not found: " * file_path)
    end
    
    file_ext = lowercase(splitext(file_path)[2])
    
    if file_ext == ".csv"
        source = CSVSource(file_path; header_row = header_row)
        return read_data(source)
    elseif file_ext in [".xlsx", ".xls"]
        source = ExcelSource(file_path; sheet = sheet, header_row = header_row)
        return read_data(source)
    else
        error("Unsupported file format: " * file_ext * ". Supported formats: .csv, .xlsx/.xls")
    end
end

# =================== UTILITY FUNCTIONS ===================

"""
    source_info(source::DataSource) -> Dict{Symbol, Any}

Returns metadata information about a data source.
"""
function source_info end

function source_info(source::CSVSource)
    return Dict{Symbol, Any}(
        :type => :csv_file,
        :path => source.path,
        :size_mb => round(stat(source.path).size / (1024^2), digits=2),
        :delimiter => source.delimiter,
        :encoding => source.encoding,
        :exists => isfile(source.path)
    )
end

function source_info(source::ExcelSource)
    return Dict{Symbol, Any}(
        :type => :excel_file,
        :path => source.path,
        :size_mb => round(stat(source.path).size / (1024^2), digits=2),
        :sheet => source.sheet,
        :range => source.range,
        :exists => isfile(source.path)
    )
end

function source_info(source::URLSource)
    return Dict{Symbol, Any}(
        :type => :url,
        :url => source.url,
        :format => source.format,
        :timeout => source.timeout,
        :headers_count => length(source.headers)
    )
end

function source_info(source::DataFrameSource)
    return Dict{Symbol, Any}(
        :type => :dataframe,
        :name => source.name,
        :rows => nrow(source.data),
        :cols => ncol(source.data),
        :size_mb => Base.summarysize(source.data) / (1024^2),
        :metadata_keys => keys(source.metadata)
    )
end

"""
    validate_source(source::DataSource) -> Bool

Validates that a data source is accessible and readable.
Returns true if valid, throws an exception with details if not.
"""
function validate_source end

function validate_source(source::FileSource)
    isfile(source.path) || throw(ArgumentError("File does not exist: $(source.path)"))
    isreadable(source.path) || throw(ArgumentError("File is not readable: $(source.path)"))
    return true
end

function validate_source(source::URLSource)
    try
        response = HTTP.head(source.url; timeout=source.timeout, headers=source.headers)
        response.status == 200 || throw(ArgumentError("URL returned status $(response.status)"))
        return true
    catch e
        throw(ArgumentError("URL validation failed for $(source.url): $e"))
    end
end

function validate_source(source::DataFrameSource)
    nrow(source.data) > 0 || throw(ArgumentError("DataFrame source is empty"))
    return true
end

# =================== CONVENIENCE CONSTRUCTORS ===================

"""
Convenience constructors that automatically choose the right data source type
based on input patterns.
"""

"""
    data_source(path_or_url::String; kwargs...) -> DataSource

Automatically creates the appropriate DataSource type based on the input string.
"""
function data_source(input::String; kwargs...)
    if startswith(input, "http")
        return URLSource(input; kwargs...)
    elseif endswith(lowercase(input), ".csv")
        return CSVSource(input; kwargs...)
    elseif endswith(lowercase(input), r"\.xlsx?$")
        return ExcelSource(input; kwargs...)
    else
        throw(ArgumentError("Cannot determine data source type for: $input"))
    end
end

"""
    data_source(df::DataFrame, name::String; kwargs...) -> DataFrameSource

Creates a DataFrameSource from an existing DataFrame.
"""
function data_source(df::DataFrame, name::String="dataframe"; kwargs...)
    return DataFrameSource(df, name; kwargs...)
end