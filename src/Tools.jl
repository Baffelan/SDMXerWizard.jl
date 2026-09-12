"""
    SDMXerWizard.Tools

Plain functions a language model calls through MCP. Every function takes a
`Session` first and keyword arguments matching the MCP argument names, and
returns a `Dict{String,Any}` holding only JSON-compatible values: strings,
numbers, booleans, `nothing`, vectors and nested dicts.

Loaded schemas and sources live in the session under short handles
(`schema_1`, `source_2`) so later calls can refer to them without re-fetching
and without passing large objects through the model's context.
"""
module Tools

using ..SDMXerWizard: SourceDataProfile, ColumnProfile, MappingCandidate, MappingConfidence,
    AdvancedMappingResult, TransformationStep, ScriptTemplate,
    profile_source_data, read_source_data, anonymize_source_data,
    create_inference_engine, infer_advanced_mappings, validate_mapping_quality,
    fuzzy_match_score, build_transformation_steps, get_loading_code, select_template
import SDMXer
using SDMXer: DataflowSchema, extract_dataflow_schema, extract_all_codelists,
    get_required_columns, get_optional_columns, create_validator, validate_sdmx_csv,
    ValidationResult, ValidationIssue
using DataFrames, CSV, JSON3, Dates

# ---------------------------------------------------------------- session

struct SchemaEntry
    schema::DataflowSchema
    codelists::Union{DataFrame, Nothing}
    origin::String
end

mutable struct SourceEntry
    data::DataFrame
    profile::SourceDataProfile
    path::String
    last_mapping::Union{AdvancedMappingResult, Nothing}
end

"""
    Session()

Holds the schemas and sources loaded so far, keyed by handle.
"""
mutable struct Session
    schemas::Dict{String, SchemaEntry}
    sources::Dict{String, SourceEntry}
    counter::Int
end
Session() = Session(Dict{String, SchemaEntry}(), Dict{String, SourceEntry}(), 0)

function next_handle!(session::Session, prefix::String)
    session.counter += 1
    return prefix * "_" * string(session.counter)
end

"""
    ToolError(message, hint)

Raised by a tool when the call cannot proceed. `run_tool` turns it into an
error dict so the calling model can read the message and the hint and retry.
"""
struct ToolError <: Exception
    message::String
    hint::String
end
Base.showerror(io::IO, e::ToolError) = print(io, e.message)

function handle_list(d::AbstractDict, first_call::String)
    isempty(d) && return "none, call " * first_call * " first"
    return join(sort(collect(keys(d))), ", ")
end

function get_schema(session::Session, id)
    key = string(id)
    haskey(session.schemas, key) && return session.schemas[key]
    throw(ToolError("Unknown schema_id " * key,
                    "Loaded schemas: " * handle_list(session.schemas, "load_schema")))
end

function get_source(session::Session, id)
    key = string(id)
    haskey(session.sources, key) && return session.sources[key]
    throw(ToolError("Unknown source_id " * key,
                    "Loaded sources: " * handle_list(session.sources, "load_source")))
end

# ---------------------------------------------------------------- dispatch

# JSON3 objects and arrays become plain Dicts and Vectors before reaching a tool.
plain(x) = x
plain(x::AbstractDict) = Dict{String, Any}(string(k) => plain(v) for (k, v) in pairs(x))
plain(x::AbstractVector) = Any[plain(v) for v in x]
plain(x::AbstractString) = String(x)

"""
    run_tool(session, f, args) -> Dict{String,Any}

Call `f(session; args...)` and turn any exception into an error dict with
`error` and `hint` keys.
"""
function run_tool(session::Session, f, args::AbstractDict)
    kwargs = Dict{Symbol, Any}(Symbol(string(k)) => plain(v) for (k, v) in pairs(args))
    try
        return f(session; kwargs...)
    catch e
        e isa ToolError && return Dict{String, Any}("error" => e.message, "hint" => e.hint)
        return Dict{String, Any}("error" => sprint(showerror, e),
                                 "hint" => "Check the arguments and call the tool again with corrected values.")
    end
end

# ---------------------------------------------------------------- serialisation

const MAX_SAMPLE_VALUES = 5
const MAX_SAMPLE_LENGTH = 60

function truncate_str(s::AbstractString, n::Integer=MAX_SAMPLE_LENGTH)
    return length(s) <= n ? String(s) : first(s, n - 1) * "…"
end

jsonable(x::Nothing) = nothing
jsonable(x::Missing) = nothing
jsonable(x::Bool) = x
jsonable(x::Integer) = Int(x)
jsonable(x::AbstractFloat) = isfinite(x) ? round(Float64(x); sigdigits=4) : nothing
jsonable(x::AbstractString) = String(x)
jsonable(x::Symbol) = string(x)
jsonable(x::Type) = string(x)
jsonable(x::Enum) = string(x)
jsonable(x::Union{Date, DateTime}) = string(x)
jsonable(x::AbstractVector) = Any[jsonable(v) for v in x]
jsonable(x::Tuple) = Any[jsonable(v) for v in x]
jsonable(x::AbstractDict) = Dict{String, Any}(string(k) => jsonable(v) for (k, v) in pairs(x))
jsonable(x::NamedTuple) = Dict{String, Any}(string(k) => jsonable(v) for (k, v) in pairs(x))
jsonable(x) = string(x)

"""
    rows(df, limit) -> Vector

The first `limit` rows of a DataFrame as dicts keyed by column name.
"""
function rows(df::DataFrame, limit::Integer)
    n = min(nrow(df), limit)
    return Any[Dict{String, Any}(string(c) => jsonable(df[i, c]) for c in names(df)) for i in 1:n]
end

function jsonable(c::ColumnProfile)
    samples = Any[truncate_str(string(v)) for v in Iterators.take(c.sample_values, MAX_SAMPLE_VALUES)]
    return Dict{String, Any}(
        "name" => c.name,
        "type" => string(c.type),
        "missing_count" => c.missing_count,
        "unique_count" => c.unique_count,
        "sample_values" => samples,
        "is_categorical" => c.is_categorical,
        "is_temporal" => c.is_temporal,
        "temporal_format" => jsonable(c.temporal_format),
        "numeric_stats" => jsonable(c.numeric_stats))
end

function jsonable(p::SourceDataProfile)
    return Dict{String, Any}(
        "path" => p.file_path,
        "file_type" => p.file_type,
        "row_count" => p.row_count,
        "column_count" => p.column_count,
        "columns" => Any[jsonable(c) for c in p.columns],
        "data_quality_score" => jsonable(p.data_quality_score),
        "suggested_key_columns" => p.suggested_key_columns,
        "suggested_value_columns" => p.suggested_value_columns,
        "suggested_time_columns" => p.suggested_time_columns)
end

# ---------------------------------------------------------------- load_source

"""
    load_source(session; path, sheet=1, header_row=1, anonymize=false)

Read a CSV or Excel file, profile it, store it in the session and return the
handle plus a compact profile. With `anonymize=true` the stored data is the
anonymised copy, so nothing downstream sees the original values.
"""
function load_source(session::Session; path, sheet=1, header_row=1, anonymize=false)
    p = String(path)
    isfile(p) || throw(ToolError("File not found: " * p,
                                 "Give an absolute path to an existing .csv, .xlsx or .xls file."))
    df = read_source_data(p; sheet=sheet, header_row=Int(header_row))
    profile = profile_source_data(df, p)
    if anonymize
        df = anonymize_source_data(df, profile)
        profile = profile_source_data(df, p)
    end
    id = next_handle!(session, "source")
    session.sources[id] = SourceEntry(df, profile, p, nothing)
    out = jsonable(profile)
    out["source_id"] = id
    out["anonymized"] = Bool(anonymize)
    return out
end

end # module Tools
