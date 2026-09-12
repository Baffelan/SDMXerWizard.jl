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

"""
    load_schema(session; url, with_codelists=true)

Fetch a dataflow structure (URL or local SDMx-ML file), store it and return a
summary. Codelists are fetched too unless `with_codelists` is false.
"""
function load_schema(session::Session; url, with_codelists=true)
    u = String(url)
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

"""
    lookup_codes(session; schema_id, dimension, query=nothing, limit=20)

List or search the codes of the codelist behind a dimension or attribute.
Without a query the first `limit` codes are returned; with one, codes whose id
or name contain the text rank first, then fuzzy matches.
"""
function lookup_codes(session::Session; schema_id, dimension, query=nothing, limit=20)
    entry = get_schema(session, schema_id)
    dim = String(dimension)
    cl = codelist_for(entry, dim)
    cl === nothing && throw(ToolError("No codelist behind " * dim,
        "Only dimensions and attributes with a codelist_id in the load_schema result can be looked up."))
    entry.codelists === nothing && throw(ToolError("Schema " * string(schema_id) * " was loaded without codelists",
        "Call load_schema again with with_codelists=true."))
    codes = codes_for(entry, cl)
    total = nrow(codes)
    scored = Tuple{Float64, DataFrameRow}[]
    if query === nothing || isempty(string(query))
        for r in eachrow(codes)
            push!(scored, (1.0, r))
        end
    else
        q = lowercase(string(query))
        for r in eachrow(codes)
            name = ismissing(r.name) ? "" : String(r.name)
            score = if occursin(q, lowercase(r.code_id)) || occursin(q, lowercase(name))
                1.0
            else
                max(fuzzy_match_score(q, r.code_id), fuzzy_match_score(q, name))
            end
            score >= 0.3 && push!(scored, (score, r))
        end
        sort!(scored; by=first, rev=true)
    end
    lim = Int(limit)
    kept = scored[1:min(lim, length(scored))]
    matches = Any[Dict{String, Any}(
        "code" => r.code_id, "name" => jsonable(r.name),
        "parent" => hasproperty(r, :parent_code_id) ? jsonable(r.parent_code_id) : nothing,
        "score" => jsonable(s)) for (s, r) in kept]
    return Dict{String, Any}("dimension" => dim, "codelist_id" => cl, "total_codes" => total,
                             "matches" => matches, "truncated" => length(scored) > lim)
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

end # module Tools
