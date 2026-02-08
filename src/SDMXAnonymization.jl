"""
AI-free source data anonymization utilities for SDMXerWizard.jl.

This module provides deterministic, target-aware anonymization to support
privacy-preserving LLM workflows. It is designed to:

- Avoid sending raw data to LLMs
- Preserve structure, types, and cardinality patterns
- Optionally align tokenization with target schema identifiers
"""

# =================== CONFIGURATION ===================

"""
    AnonymizationConfig

Configuration options for deterministic anonymization.

# Fields
- `mode::Symbol`: `:safe` (default) or `:preserve_distribution`
- `max_unique_categorical::Int`: threshold for categorical tokenization
- `max_rows::Int`: maximum rows to return (sampling)
- `preserve_missing::Bool`: keep `missing` values as-is
- `prefix_with_target::Bool`: prefix tokens with matched target IDs
"""
struct AnonymizationConfig
    mode::Symbol
    max_unique_categorical::Int
    max_rows::Int
    preserve_missing::Bool
    prefix_with_target::Bool
end

AnonymizationConfig() = AnonymizationConfig(:safe, 1000, 50, true, true)

# =================== PUBLIC API ===================

"""
    anonymize_source_data(
        data::DataFrame,
        profile::Union{SourceDataProfile, Nothing}=nothing;
        target_schema::Union{DataflowSchema, Nothing}=nothing,
        mappings::Dict{String, String}=Dict{String, String}(),
        config::AnonymizationConfig=AnonymizationConfig()
    ) -> DataFrame

Deterministically anonymizes source data. If `profile` is not provided, it will
infer minimal column characteristics from `data`.

If `target_schema` or `mappings` are provided, token prefixes may align with
target column IDs (e.g., GEO_PICT_001).

This is AI-free and safe to run on sensitive data.
"""
function anonymize_source_data(
    data::DataFrame,
    profile::Union{SourceDataProfile, Nothing}=nothing;
    target_schema::Union{DataflowSchema, Nothing}=nothing,
    mappings::Dict{String, String}=Dict{String, String}(),
    config::AnonymizationConfig=AnonymizationConfig()
)
    rows = nrow(data)
    if rows > config.max_rows
        data = data[1:config.max_rows, :]
    end

    # Build a quick profile if not provided
    profile = profile === nothing ? profile_source_data(data) : profile

    anonymized = DataFrame()
    target_ids = _collect_target_ids(target_schema)

    for col_profile in profile.columns
        col_name = col_profile.name
        if !(col_name in names(data))
            continue
        end

        raw = data[!, col_name]
        target_hint = _resolve_target_hint(col_name, mappings, target_ids)
        anonymized[!, col_name] = anonymize_column_values(
            raw,
            col_profile;
            target_hint=target_hint,
            config=config
        )
    end

    return anonymized
end

"""
    anonymize_column_values(
        column::AbstractVector,
        profile::ColumnProfile;
        target_hint::Union{String, Nothing}=nothing,
        config::AnonymizationConfig=AnonymizationConfig()
    ) -> Vector

Anonymizes a single column with deterministic tokenization.
"""
function anonymize_column_values(
    column::AbstractVector,
    profile::ColumnProfile;
    target_hint::Union{String, Nothing}=nothing,
    config::AnonymizationConfig=AnonymizationConfig()
)
    if profile.is_temporal
        return _anonymize_temporal_column(column, profile, target_hint, config)
    elseif profile.numeric_stats !== nothing && !profile.is_categorical
        return _anonymize_numeric_column(column, target_hint, config)
    elseif profile.is_categorical && profile.unique_count <= config.max_unique_categorical
        return _anonymize_categorical_column(column, target_hint, config)
    else
        return _anonymize_text_column(column, target_hint, config)
    end
end

"""
    summarize_anonymized_data(data::DataFrame; max_samples::Int=5) -> NamedTuple

Builds a compact summary of anonymized data, safe to pass to LLMs.
"""
function summarize_anonymized_data(data::DataFrame; max_samples::Int=5)
    cols = names(data)
    return (
        columns = cols,
        types = eltype.(eachcol(data)),
        sample_values = NamedTuple(
            Symbol(col) => _anonymize_sample_unique(data[!, col], max_samples)
            for col in cols
        ),
        dimensions = size(data)
    )
end

# =================== INTERNAL HELPERS ===================

function _anonymize_sample_unique(col::AbstractVector, n::Int=5)
    vals = unique(col)
    return vals[1:min(n, length(vals))]
end

function _collect_target_ids(target_schema::Union{DataflowSchema, Nothing})
    target_schema === nothing && return String[]
    ids = String[]
    append!(ids, target_schema.dimensions.dimension_id)
    append!(ids, target_schema.measures.measure_id)
    append!(ids, target_schema.attributes.attribute_id)
    if target_schema.time_dimension !== nothing
        push!(ids, target_schema.time_dimension.dimension_id)
    end
    return ids
end

function _resolve_target_hint(
    source_col::String,
    mappings::Dict{String, String},
    target_ids::Vector{String}
)
    if haskey(mappings, source_col)
        return mappings[source_col]
    end

    source_norm = _normalize_name(source_col)
    matches = String[]
    for target_id in target_ids
        target_norm = _normalize_name(target_id)
        if occursin(target_norm, source_norm) || occursin(source_norm, target_norm)
            push!(matches, target_id)
        end
    end

    return length(matches) == 1 ? matches[1] : nothing
end

function _normalize_name(s::String)
    return replace(lowercase(s), r"[^a-z0-9]" => "")
end

function _token_prefix(token_type::String, target_hint::Union{String, Nothing}, config::AnonymizationConfig)
    if config.prefix_with_target && target_hint !== nothing
        return target_hint
    end
    return token_type
end

function _tokenize_values(values::AbstractVector, token_type::String, target_hint, config::AnonymizationConfig)
    prefix = _token_prefix(token_type, target_hint, config)
    mapping = Dict{Any, String}()
    result = Vector{Any}(undef, length(values))
    counter = 0

    for (i, v) in enumerate(values)
        if ismissing(v) && config.preserve_missing
            result[i] = missing
            continue
        end

        if !haskey(mapping, v)
            counter += 1
            mapping[v] = "$(prefix)_$(lpad(string(counter), 3, '0'))"
        end
        result[i] = mapping[v]
    end

    return result
end

function _anonymize_categorical_column(column, target_hint, config::AnonymizationConfig)
    return _tokenize_values(column, "CAT", target_hint, config)
end

function _anonymize_text_column(column, target_hint, config::AnonymizationConfig)
    prefix = _token_prefix("TXT", target_hint, config)
    mapping = Dict{Any, String}()
    result = Vector{Any}(undef, length(column))
    counter = 0

    for (i, v) in enumerate(column)
        if ismissing(v) && config.preserve_missing
            result[i] = missing
            continue
        end

        if !haskey(mapping, v)
            counter += 1
            len = length(string(v))
            mapping[v] = "$(prefix)_$(lpad(string(counter), 3, '0'))_L$(len)"
        end
        result[i] = mapping[v]
    end

    return result
end

function _anonymize_numeric_column(column, target_hint, config::AnonymizationConfig)
    non_missing = filter(!ismissing, column)
    unique_vals = unique(non_missing)
    sorted_vals = sort(unique_vals)

    mapping = Dict{Any, String}()
    prefix = _token_prefix("NUM", target_hint, config)

    if config.mode == :preserve_distribution && !isempty(sorted_vals)
        bucket_count = 10
        for (idx, v) in enumerate(sorted_vals)
            pct = idx / length(sorted_vals)
            bucket = max(1, min(bucket_count, ceil(Int, pct * bucket_count)))
            mapping[v] = "$(prefix)_Q$(bucket)"
        end
    else
        for (idx, v) in enumerate(sorted_vals)
            mapping[v] = "$(prefix)_$(lpad(string(idx), 3, '0'))"
        end
    end

    result = Vector{Any}(undef, length(column))
    for (i, v) in enumerate(column)
        if ismissing(v) && config.preserve_missing
            result[i] = missing
        else
            result[i] = mapping[v]
        end
    end

    return result
end

function _anonymize_temporal_column(column, profile::ColumnProfile, target_hint, config::AnonymizationConfig)
    prefix = _token_prefix("TIME", target_hint, config)
    result = Vector{Any}(undef, length(column))

    for (i, v) in enumerate(column)
        if ismissing(v) && config.preserve_missing
            result[i] = missing
            continue
        end
        result[i] = _anonymize_temporal_value(v, profile.temporal_format, prefix)
    end

    return result
end

function _anonymize_temporal_value(v, temporal_format::Union{String, Nothing}, prefix::String)
    if v isa Date
        return "$(prefix)_$(year(v))"
    elseif v isa DateTime
        return "$(prefix)_$(year(v))"
    end

    s = string(v)
    if temporal_format == "YYYY"
        return "$(prefix)_$(s)"
    elseif temporal_format == "YYYY-MM"
        return "$(prefix)_$(s[1:4])_M"
    elseif temporal_format == "YYYY-MM-DD"
        return "$(prefix)_$(s[1:4])_D"
    elseif temporal_format == "YYYY-Q"
        return "$(prefix)_$(s)"
    end

    # Fallback: year-like tokens if parseable
    m = match(r"(\d{4})", s)
    if m !== nothing
        return "$(prefix)_$(m.captures[1])"
    end

    return "$(prefix)_UNK"
end
