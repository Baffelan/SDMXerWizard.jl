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
    get_required_columns, get_optional_columns, get_dimension_order, create_validator, validate_sdmx_csv,
    ValidationResult, ValidationIssue, construct_data_url, fetch_sdmx_data
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
    last_result::Union{DataFrame, Nothing}
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
    session.sources[id] = SourceEntry(df, profile, p, nothing, nothing)
    out = jsonable(profile)
    out["source_id"] = id
    out["anonymized"] = Bool(anonymize)
    return out
end

# ---------------------------------------------------------------- schemas

"""
    register_schema!(session, schema, codelists; origin="") -> String

Store a schema built elsewhere (tests, programmatic use) and return its handle.
"""
function register_schema!(session::Session, schema::DataflowSchema,
                          codelists::Union{DataFrame, Nothing}; origin::String="")
    id = next_handle!(session, "schema")
    session.schemas[id] = SchemaEntry(schema, codelists, origin)
    return id
end

"""
    codelist_for(entry, column_id) -> Union{String, Nothing}

The codelist id behind a dimension or attribute, or `nothing`.
"""
function codelist_for(entry::SchemaEntry, column_id::AbstractString)
    schema = entry.schema
    for row in eachrow(schema.dimensions)
        if row.dimension_id == column_id
            return ismissing(row.codelist_id) ? nothing : String(row.codelist_id)
        end
    end
    for row in eachrow(schema.attributes)
        if row.attribute_id == column_id
            return ismissing(row.codelist_id) ? nothing : String(row.codelist_id)
        end
    end
    return nothing
end

function codes_for(entry::SchemaEntry, codelist_id::AbstractString)
    entry.codelists === nothing && return DataFrame()
    return entry.codelists[entry.codelists.codelist_id .== codelist_id, :]
end

function field_or_nothing(nt, key::Symbol)
    return hasproperty(nt, key) ? jsonable(getproperty(nt, key)) : nothing
end

"""
    schema_summary(entry, id) -> Dict

Compact description of a loaded schema: dimensions in order, the time
dimension, attributes, measures, required and optional columns, and the size
of each codelist. Codes themselves are never included.
"""
function schema_summary(entry::SchemaEntry, id::String)
    schema = entry.schema
    info = schema.dataflow_info
    dims = Any[Dict{String, Any}(
        "id" => r.dimension_id, "position" => r.position,
        "codelist_id" => jsonable(r.codelist_id), "is_time_dimension" => r.is_time_dimension)
        for r in eachrow(schema.dimensions)]
    td = schema.time_dimension
    time_dim = td === nothing ? nothing : Dict{String, Any}(
        "id" => td.dimension_id, "position" => td.position, "data_type" => jsonable(td.data_type))
    attrs = Any[Dict{String, Any}(
        "id" => r.attribute_id, "assignment_status" => jsonable(r.assignment_status),
        "relationship" => jsonable(r.relationship), "codelist_id" => jsonable(r.codelist_id))
        for r in eachrow(schema.attributes)]
    measures = Any[Dict{String, Any}("id" => r.measure_id, "data_type" => jsonable(r.data_type))
        for r in eachrow(schema.measures)]
    counts = Dict{String, Any}()
    if entry.codelists !== nothing && nrow(entry.codelists) > 0
        for g in groupby(entry.codelists, :codelist_id)
            counts[String(first(g.codelist_id))] = nrow(g)
        end
    end
    return Dict{String, Any}(
        "schema_id" => id,
        "origin" => entry.origin,
        "dataflow" => Dict{String, Any}(
            "id" => field_or_nothing(info, :id), "agency" => field_or_nothing(info, :agency),
            "version" => field_or_nothing(info, :version), "name" => field_or_nothing(info, :name)),
        "dimensions" => dims,
        "time_dimension" => time_dim,
        "attributes" => attrs,
        "measures" => measures,
        "required_columns" => get_required_columns(schema),
        "optional_columns" => get_optional_columns(schema),
        "codelists" => counts,
        "codelists_loaded" => entry.codelists !== nothing)
end

# SDMx REST providers, keyed like the SDMx MCP gateway
# (https://github.com/Baffelan/sdmx-mcp-gateway) so ids read off its results
# pass straight through. `agency` is the default owner of dataflows there.
const PROVIDERS = Dict{String, NamedTuple{(:base_url, :agency), Tuple{String, String}}}(
    "SPC" => (base_url = "https://stats-sdmx-disseminate.pacificdata.org/rest", agency = "SPC"),
    "FBOS" => (base_url = "https://data-sdmx-disseminate.statsfiji.gov.fj/rest", agency = "FBOS"),
    "SBS" => (base_url = "https://data-sdmx-disseminate.sbs.gov.ws/rest", agency = "SBS"),
    "ECB" => (base_url = "https://data-api.ecb.europa.eu/service", agency = "ECB"),
    "UNICEF" => (base_url = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest", agency = "UNICEF"),
    "IMF" => (base_url = "https://api.imf.org/external/sdmx/2.1", agency = "IMF.STA"),
    "OECD" => (base_url = "https://sdmx.oecd.org/public/rest", agency = "OECD"),
    "ESTAT" => (base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1", agency = "ESTAT"),
    "ILO" => (base_url = "https://sdmx.ilo.org/rest", agency = "ILO"),
    "ABS" => (base_url = "https://data.api.abs.gov.au/rest", agency = "ABS"),
    "BIS" => (base_url = "https://stats.bis.org/api/v1", agency = "BIS"),
    "STATSNZ" => (base_url = "https://api.data.stats.govt.nz/rest", agency = "STATSNZ"))

"""
    structure_url(endpoint, dataflow_id; agency=nothing, version="latest") -> String

The structure URL (`references=all`) of a dataflow on one of the known
providers. `agency` defaults to the provider's own agency id; OECD dataflows
live under sub-agencies, so pass the agency the gateway reports.
"""
function structure_url(endpoint, dataflow_id; agency=nothing, version="latest")
    key = uppercase(strip(string(endpoint)))
    haskey(PROVIDERS, key) || throw(ToolError("Unknown endpoint " * key,
        "Known endpoints: " * join(sort(collect(keys(PROVIDERS))), ", ") * ". Pass url instead for any other provider."))
    p = PROVIDERS[key]
    ag = agency === nothing || isempty(string(agency)) ? p.agency : string(agency)
    ver = version === nothing || isempty(string(version)) ? "latest" : string(version)
    return p.base_url * "/dataflow/" * ag * "/" * string(dataflow_id) * "/" * ver * "?references=all"
end

"""
    load_schema(session; url=nothing, endpoint=nothing, dataflow_id=nothing, agency=nothing,
                version="latest", with_codelists=true)

Fetch a dataflow structure, store it and return a summary. Give either a
`url` (a .Stat developer API link, or a local SDMx-ML file) or an `endpoint`
key plus `dataflow_id` as reported by the SDMx MCP gateway. Codelists are
fetched too unless `with_codelists` is false.
"""
function load_schema(session::Session; url=nothing, endpoint=nothing, dataflow_id=nothing,
                     agency=nothing, version="latest", with_codelists=true)
    u = if url !== nothing && !isempty(string(url))
        string(url)
    elseif endpoint !== nothing && dataflow_id !== nothing
        structure_url(endpoint, dataflow_id; agency=agency, version=version)
    else
        throw(ToolError("Give either url, or endpoint and dataflow_id",
            "Use list_dataflows on the sdmx-gateway server to find the dataflow, then pass its endpoint, agency and id here."))
    end
    schema = extract_dataflow_schema(u)
    codelists = nothing
    if with_codelists
        try
            codelists = extract_all_codelists(u)
        catch e
            @warn "Codelists could not be fetched" url=u exception=e
        end
    end
    id = register_schema!(session, schema, codelists; origin=u)
    return schema_summary(session.schemas[id], id)
end


# ---------------------------------------------------------------- mappings

const MAPPING_METHODS = ("heuristic", "fuzzy", "advanced")

function jsonable(m::MappingCandidate)
    return Dict{String, Any}(
        "source_column" => m.source_column,
        "target_column" => m.target_column,
        "confidence" => jsonable(m.confidence_score),
        "confidence_level" => string(m.confidence_level),
        "match_type" => m.match_type,
        "evidence" => jsonable(m.evidence),
        "suggested_transformation" => jsonable(m.suggested_transformation),
        "notes" => m.validation_notes)
end

# Give the engine the codelists already held in the session so it never fetches.
function preload_codelists!(engine, entry::SchemaEntry)
    entry.codelists === nothing && return engine
    for g in groupby(entry.codelists, :codelist_id)
        engine.codelists_data[String(first(g.codelist_id))] = DataFrame(g)
    end
    return engine
end

function target_order(schema::DataflowSchema)
    order = String[]
    append!(order, schema.dimensions.dimension_id)
    schema.time_dimension === nothing || push!(order, schema.time_dimension.dimension_id)
    append!(order, schema.measures.measure_id)
    append!(order, schema.attributes.attribute_id)
    return order
end

"""
    mapping_report(result, entry) -> Dict

Candidates grouped by target column in schema order, plus the unmapped
columns and the warnings from `validate_mapping_quality`.
"""
function mapping_report(result::AdvancedMappingResult, entry::SchemaEntry)
    schema = entry.schema
    grouped = Dict{String, Vector{MappingCandidate}}()
    for m in result.mappings
        push!(get!(grouped, m.target_column, MappingCandidate[]), m)
    end
    mappings = Any[]
    for t in target_order(schema)
        haskey(grouped, t) || continue
        cands = sort(grouped[t]; by=c -> c.confidence_score, rev=true)
        push!(mappings, Dict{String, Any}("target_column" => t,
            "codelist_id" => jsonable(codelist_for(entry, t)),
            "candidates" => Any[jsonable(c) for c in cands]))
    end
    quality = validate_mapping_quality(result, schema)
    return Dict{String, Any}(
        "mappings" => mappings,
        "unmapped_source_columns" => result.unmapped_source_columns,
        "unmapped_target_columns" => result.unmapped_target_columns,
        "unmapped_required_columns" => intersect(result.unmapped_target_columns, get_required_columns(schema)),
        "quality_score" => jsonable(result.quality_score),
        "recommendations" => result.recommendations,
        "warnings" => vcat(quality["critical_issues"], quality["warnings"]))
end

"""
    infer_mappings(session; source_id, schema_id, method="advanced", confidence_threshold=0.3)

Rank source columns against the target columns of a schema. `heuristic` uses
column names only, `fuzzy` adds statistical analysis, `advanced` adds codelist
value matching. The result is kept on the source for `transformation_plan`.
"""
function infer_mappings(session::Session; source_id, schema_id, method="advanced", confidence_threshold=0.3)
    m = String(method)
    m in MAPPING_METHODS || throw(ToolError("Unknown method " * m,
        "method must be one of: " * join(MAPPING_METHODS, ", ")))
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    engine = create_inference_engine(min_confidence=Float64(confidence_threshold),
        use_statistical_analysis=(m != "heuristic"), use_value_matching=(m == "advanced"))
    m == "advanced" && preload_codelists!(engine, entry)
    result = infer_advanced_mappings(engine, src.profile, entry.schema, src.data)
    src.last_mapping = result
    out = mapping_report(result, entry)
    out["method"] = m
    out["source_id"] = String(source_id)
    out["schema_id"] = String(schema_id)
    return out
end

# ---------------------------------------------------------------- plan

const SCRIPT_CONTRACT = """
The script is evaluated in a fresh module where `source` is the loaded source DataFrame \
and the packages DataFrames, CSV and Dates are already in scope, together with SDMXer. \
Do not read the file again. The script must assign a DataFrame to a variable named \
`result` whose columns are the SDMx columns of the target schema: dimensions in order, \
TIME_PERIOD, OBS_VALUE, then attributes. Code values must be codelist ids, never labels. \
Use string concatenation rather than string interpolation."""

function jsonable(s::TransformationStep)
    return Dict{String, Any}(
        "name" => s.step_name, "operation_type" => s.operation_type,
        "function" => s.tidier_function, "description" => s.description,
        "source_columns" => s.source_columns, "target_columns" => s.target_columns,
        "logic" => s.transformation_logic, "validation_checks" => s.validation_checks,
        "comments" => s.comments)
end

"""
    choose_mappings(result, overrides, entry, src) -> AdvancedMappingResult

One candidate per target column: the override when given, otherwise the
highest-confidence inferred candidate.
"""
function choose_mappings(result::AdvancedMappingResult, overrides::Dict{String, String},
                         entry::SchemaEntry, src::SourceEntry)
    best = Dict{String, MappingCandidate}()
    for m in result.mappings
        if !haskey(best, m.target_column) || m.confidence_score > best[m.target_column].confidence_score
            best[m.target_column] = m
        end
    end
    source_names = names(src.data)
    for (target, source) in overrides
        source in source_names || throw(ToolError("Source column " * source * " does not exist",
            "Source columns: " * join(source_names, ", ")))
        best[target] = MappingCandidate(source, target, 1.0, MappingConfidence(5), "user",
            Dict{String, Any}("origin" => "override"), nothing, String[])
    end
    order = target_order(entry.schema)
    chosen = MappingCandidate[best[t] for t in order if haskey(best, t)]
    mapped_sources = Set(m.source_column for m in chosen)
    mapped_targets = Set(m.target_column for m in chosen)
    return AdvancedMappingResult(chosen, result.coverage_analysis,
        String[c for c in source_names if !(c in mapped_sources)],
        String[t for t in order if !(t in mapped_targets)],
        result.quality_score, result.recommendations, result.transformation_complexity)
end

const MAX_RECODE_VALUES = 50

function code_candidates(value::String, codes::DataFrame)
    lv = lowercase(value)
    scored = Tuple{Float64, DataFrameRow}[]
    for r in eachrow(codes)
        name = ismissing(r.name) ? "" : String(r.name)
        contained = occursin(lv, lowercase(r.code_id)) || (!isempty(name) && occursin(lv, lowercase(name)))
        score = contained ? 1.0 : max(fuzzy_match_score(value, r.code_id),
                                      isempty(name) ? 0.0 : fuzzy_match_score(value, name))
        push!(scored, (score, r))
    end
    sort!(scored; by=first, rev=true)
    return Any[Dict{String, Any}("code" => r.code_id, "name" => jsonable(r.name), "score" => jsonable(s))
               for (s, r) in Iterators.take(scored, 3) if s > 0.0]
end

"""
    recodings(entry, chosen, data) -> Vector

For every chosen mapping whose target has a codelist, the distinct source
values that are not already code ids, each with up to three candidate codes.
"""
function recodings(entry::SchemaEntry, chosen::AdvancedMappingResult, data::DataFrame)
    out = Any[]
    entry.codelists === nothing && return out
    for m in chosen.mappings
        cl = codelist_for(entry, m.target_column)
        cl === nothing && continue
        codes = codes_for(entry, cl)
        nrow(codes) == 0 && continue
        ids = Set(String.(codes.code_id))
        values = unique(String[string(v) for v in skipmissing(data[!, m.source_column])])
        pending = String[v for v in values if !(v in ids)]
        isempty(pending) && continue
        entries = Any[Dict{String, Any}("source_value" => v, "candidates" => code_candidates(v, codes))
                      for v in Iterators.take(pending, MAX_RECODE_VALUES)]
        push!(out, Dict{String, Any}("target_column" => m.target_column, "source_column" => m.source_column,
            "codelist_id" => cl, "values" => entries, "truncated" => length(pending) > MAX_RECODE_VALUES))
    end
    return out
end

"""
    transformation_plan(session; source_id, schema_id, mappings=nothing)

Turn the chosen mappings into ordered transformation steps, a loading snippet,
a template skeleton, the recodings still to decide, and the contract a script
must follow. `mappings` optionally overrides the inferred candidates as an
object `{target_column: source_column}`. Runs `infer_mappings` with defaults
if it has not run for this source yet.
"""
function transformation_plan(session::Session; source_id, schema_id, mappings=nothing)
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    if src.last_mapping === nothing
        infer_mappings(session; source_id=source_id, schema_id=schema_id)
    end
    overrides = Dict{String, String}()
    if mappings !== nothing
        mappings isa AbstractDict || throw(ToolError("mappings must be an object",
            "Use the form {\"TARGET_COLUMN\": \"source column\"}."))
        for (k, v) in pairs(mappings)
            overrides[string(k)] = string(v)
        end
    end
    chosen = choose_mappings(src.last_mapping, overrides, entry, src)
    steps = build_transformation_steps(chosen, src.profile, entry.schema)
    template = select_template(src.profile)
    return Dict{String, Any}(
        "source_id" => String(source_id), "schema_id" => String(schema_id),
        "chosen_mappings" => Any[jsonable(m) for m in chosen.mappings],
        "unmapped_target_columns" => chosen.unmapped_target_columns,
        "unmapped_required_columns" => intersect(chosen.unmapped_target_columns, get_required_columns(entry.schema)),
        "steps" => Any[jsonable(s) for s in steps],
        "loading_code" => get_loading_code(src.profile),
        "template" => Dict{String, Any}("name" => template.template_name, "description" => template.description,
            "sections" => jsonable(template.template_sections), "example" => template.example_code),
        "recodings" => recodings(entry, chosen, src.data),
        "contract" => SCRIPT_CONTRACT)
end

# ---------------------------------------------------------------- validation

function jsonable(i::ValidationIssue)
    return Dict{String, Any}(
        "rule_id" => i.rule_id, "severity" => string(i.severity), "message" => i.message,
        "location" => i.location, "affected_row_count" => length(i.affected_rows),
        "affected_rows" => collect(Iterators.take(i.affected_rows, 10)),
        "suggested_fix" => i.suggested_fix, "auto_fixable" => i.auto_fixable)
end

function jsonable(v::ValidationResult)
    return Dict{String, Any}(
        "dataset_name" => v.dataset_name, "compliance_status" => v.compliance_status,
        "overall_score" => jsonable(v.overall_score), "total_rows" => v.total_rows,
        "total_columns" => v.total_columns, "issues" => Any[jsonable(i) for i in v.issues],
        "recommendations" => v.recommendations, "statistics" => jsonable(v.statistics))
end

function validation_report(schema::DataflowSchema, df::DataFrame, name::String, preview_rows::Integer)
    validator = create_validator(schema)
    vr = validate_sdmx_csv(validator, df, name)
    return Dict{String, Any}("validation" => jsonable(vr), "preview" => rows(df, preview_rows),
                             "result_rows" => nrow(df), "result_columns" => names(df))
end

# ---------------------------------------------------------------- sandbox

const SANDBOX_MODULES = (DataFrames, CSV, Dates, SDMXer)

function bind_module!(m::Module, pkg::Module)
    Core.eval(m, Expr(:const, Expr(:(=), nameof(pkg), pkg)))
    Core.eval(m, Expr(:using, Expr(:., :., nameof(pkg))))
    return m
end

"""
    sandbox_run(script, source) -> DataFrame

Evaluate `script` in a fresh module with `source` bound to a copy of the
DataFrame and DataFrames, CSV, Dates and SDMXer in scope. Returns the
DataFrame the script assigned to `result`.
"""
function sandbox_run(script::String, source::DataFrame)
    m = Module(:SDMXerScript)
    for pkg in SANDBOX_MODULES
        bind_module!(m, pkg)
    end
    Core.eval(m, Expr(:(=), :source, copy(source)))
    parsed = try
        Meta.parseall(script; filename="script.jl")
    catch e
        throw(ToolError("Script failed to parse: " * sprint(showerror, e), SCRIPT_CONTRACT))
    end
    line = 0
    for ex in parsed.args
        if ex isa LineNumberNode
            line = ex.line
            continue
        end
        if ex isa Expr && (ex.head == :error || ex.head == :incomplete)
            throw(ToolError("Script failed to parse at line " * string(line) * ": " * string(ex.args[1]),
                            SCRIPT_CONTRACT))
        end
        try
            Core.eval(m, ex)
        catch e
            throw(ToolError("Script failed at line " * string(line) * ": " * sprint(showerror, e),
                            SCRIPT_CONTRACT))
        end
    end
    # Bindings created by eval live in a newer world than this function, so the
    # lookups must run in the latest world (Julia 1.12 binding partitions).
    Base.invokelatest(isdefined, m, :result) || throw(ToolError("The script did not define `result`", SCRIPT_CONTRACT))
    result = Base.invokelatest(getfield, m, :result)
    result isa DataFrame || throw(ToolError("`result` is a " * string(typeof(result)) * ", expected a DataFrame",
                                            SCRIPT_CONTRACT))
    return result
end

"""
    run_script(session; source_id, schema_id, script, output_path=nothing, preview_rows=10)

Evaluate a transformation script against a loaded source, validate the
resulting DataFrame against the schema, and optionally write it as CSV.
Script errors come back as an error dict with the failing line.
"""
function run_script(session::Session; source_id, schema_id, script, output_path=nothing, preview_rows=10)
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    result = sandbox_run(String(script), src.data)
    src.last_result = result
    out = validation_report(entry.schema, result, basename(src.path), Int(preview_rows))
    if output_path !== nothing && !isempty(string(output_path))
        CSV.write(String(output_path), result)
        out["output_path"] = String(output_path)
    else
        out["output_path"] = nothing
    end
    out["source_id"] = String(source_id)
    out["schema_id"] = String(schema_id)
    return out
end

# Dimension, time and attribute columns are codes and periods: keep them as
# strings so "2015" or "01" survive the round trip through CSV.
function read_sdmx_csv(path::String, schema::DataflowSchema)
    coded = Set{String}(schema.dimensions.dimension_id)
    schema.time_dimension === nothing || push!(coded, schema.time_dimension.dimension_id)
    union!(coded, schema.attributes.attribute_id)
    return CSV.read(path, DataFrame; types=(i, name) -> String(name) in coded ? String : nothing)
end

"""
    validate_csv(session; path, schema_id, preview_rows=10)

Validate an SDMx-CSV file produced elsewhere against a loaded schema.
"""
function validate_csv(session::Session; path, schema_id, preview_rows=10)
    p = String(path)
    isfile(p) || throw(ToolError("File not found: " * p, "Give the path of an existing CSV file."))
    entry = get_schema(session, schema_id)
    df = read_sdmx_csv(p, entry.schema)
    out = validation_report(entry.schema, df, basename(p), Int(preview_rows))
    out["path"] = p
    out["schema_id"] = String(schema_id)
    return out
end

# ---------------------------------------------------------------- cross-check

# base URL, agency, dataflow id and version from a structure URL such as
# https://host/rest/dataflow/AGENCY/DF_ID/latest?references=all
function parse_structure_url(origin::String)
    m = match(r"^(.*?)/dataflow/([^/]+)/([^/]+)/([^/?]+)", origin)
    m === nothing && throw(ToolError("Cannot derive a data URL from " * origin,
        "The schema must come from a dataflow URL, or pass data_url built with the gateway's build_data_url."))
    return (base_url = String(m.captures[1]), agency = String(m.captures[2]),
            dataflow_id = String(m.captures[3]), version = String(m.captures[4]))
end

const MAX_MISMATCH_SAMPLES = 10
const MAX_KEY_SAMPLES = 5

as_float(x) = x === missing ? nothing : (x isa Number ? Float64(x) : tryparse(Float64, string(x)))

"""
    compare_frames(schema, ours, published) -> Dict

Row-by-row comparison of a transformed result with a published dataset on the
dimension and time columns they share: coverage on both sides and OBS_VALUE
agreement where the keys match.
"""
function compare_frames(schema::DataflowSchema, ours::DataFrame, published::DataFrame)
    key_cols = String[]
    for c in get_dimension_order(schema)
        c in names(ours) && c in names(published) && push!(key_cols, String(c))
    end
    td = schema.time_dimension
    if td !== nothing && td.dimension_id in names(ours) && td.dimension_id in names(published)
        push!(key_cols, String(td.dimension_id))
    end
    isempty(key_cols) && throw(ToolError("No dimension columns in common between the result and the published data",
        "The result must use the schema's dimension ids as column names."))
    ("OBS_VALUE" in names(ours) && "OBS_VALUE" in names(published)) ||
        throw(ToolError("OBS_VALUE is missing on one side", "Both frames need an OBS_VALUE column."))
    keyof(row) = join((string(row[c]) for c in key_cols), "|")
    ours_map = Dict{String, Any}(keyof(r) => as_float(r["OBS_VALUE"]) for r in eachrow(ours))
    pub_map = Dict{String, Any}(keyof(r) => as_float(r["OBS_VALUE"]) for r in eachrow(published))
    matched = String[k for k in keys(ours_map) if haskey(pub_map, k)]
    only_ours = String[k for k in keys(ours_map) if !haskey(pub_map, k)]
    only_pub = String[k for k in keys(pub_map) if !haskey(ours_map, k)]
    exact = 0; close = 0; max_rel = 0.0
    mismatches = Any[]
    for k in sort(matched)
        a = ours_map[k]; b = pub_map[k]
        if a === nothing || b === nothing
            push!(mismatches, Dict{String, Any}("key" => k, "ours" => a, "published" => b))
            continue
        end
        if a == b
            exact += 1
        else
            rel = abs(a - b) / max(abs(b), 1e-12)
            max_rel = max(max_rel, rel)
            if rel <= 0.01
                close += 1
            elseif length(mismatches) < MAX_MISMATCH_SAMPLES
                push!(mismatches, Dict{String, Any}("key" => k, "ours" => a, "published" => b,
                                                    "relative_difference" => jsonable(rel)))
            end
        end
    end
    return Dict{String, Any}(
        "key_columns" => key_cols,
        "our_rows" => nrow(ours), "published_rows" => nrow(published),
        "matched" => length(matched), "only_in_ours" => length(only_ours), "only_in_published" => length(only_pub),
        "only_in_ours_sample" => collect(Iterators.take(sort(only_ours), MAX_KEY_SAMPLES)),
        "only_in_published_sample" => collect(Iterators.take(sort(only_pub), MAX_KEY_SAMPLES)),
        "exact_agreement" => exact, "within_one_percent" => close,
        "disagreements" => length(matched) - exact - close,
        "max_relative_difference" => jsonable(max_rel),
        "disagreement_sample" => mismatches)
end

"""
    compare_with_published(session; source_id, schema_id, data_url=nothing, filters=nothing,
                           start_period=nothing, end_period=nothing)

Fetch what the provider already publishes for this dataflow and compare it
with the last `run_script` result on the shared dimension and time columns.
Give `data_url` (for example from the gateway's build_data_url), or
`filters` such as `{"REF_AREA": "KOR"}` and the URL is built from the schema.
"""
function compare_with_published(session::Session; source_id, schema_id, data_url=nothing, filters=nothing,
                                start_period=nothing, end_period=nothing)
    src = get_source(session, source_id)
    entry = get_schema(session, schema_id)
    src.last_result === nothing && throw(ToolError("No result to compare for " * string(source_id),
        "Call run_script first; its result is what gets compared."))
    url = if data_url !== nothing && !isempty(string(data_url))
        string(data_url)
    else
        parts = parse_structure_url(entry.origin)
        # Data queries want the resolved version; "latest" is only valid for structure queries.
        version = parts.version
        info = entry.schema.dataflow_info
        if version in ("latest", "all", "+") && hasproperty(info, :version) && !ismissing(info.version)
            version = string(info.version)
        end
        dimension_filters = Dict{String, Any}()
        if filters !== nothing
            filters isa AbstractDict || throw(ToolError("filters must be an object", "Use {\"REF_AREA\": \"KOR\"}."))
            for (k, v) in pairs(filters)
                dimension_filters[string(k)] = string(v)
            end
        end
        construct_data_url(parts.base_url, parts.agency, parts.dataflow_id, version;
            schema=entry.schema, dimension_filters=dimension_filters,
            start_period=start_period === nothing ? nothing : string(start_period),
            end_period=end_period === nothing ? nothing : string(end_period))
    end
    published = fetch_sdmx_data(url; timeout=120)
    nrow(published) == 0 && throw(ToolError("The provider returned no data for " * url,
        "Loosen the filters or check the key with the gateway's get_data_availability."))
    out = compare_frames(entry.schema, src.last_result, published)
    out["data_url"] = url
    out["source_id"] = String(source_id)
    out["schema_id"] = String(schema_id)
    return out
end

end # module Tools
