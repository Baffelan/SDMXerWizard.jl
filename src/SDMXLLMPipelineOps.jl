"""
LLM-specific Pipeline Operators for SDMXerWizard.jl
"""

# Dependencies loaded at package level


"""
    llm_map_with(engine::InferenceEngine, schema::DataflowSchema) -> Function

Creates an LLM-enhanced mapping function that can be used in pipelines.

# Example
```julia
mappings = profile |> llm_map_with(engine, schema)
```
"""
function llm_map_with(engine::InferenceEngine, schema::DataflowSchema)
    return profile -> infer_advanced_mappings(engine, profile, schema)
end

"""
    llm_generate_with(generator::ScriptGenerator, schema::DataflowSchema; kwargs...) -> Function

Creates an LLM-powered script generation function that can be used in pipelines.

# Example
```julia
script = mappings |> llm_generate_with(generator, schema)
```
"""
function llm_generate_with(generator::ScriptGenerator, schema::DataflowSchema; kwargs...)
    return mappings -> begin
        # We need a profile for script generation, so this is a bit more complex
        # This function expects a tuple (profile, mappings) as input
        if isa(mappings, Tuple)
            profile, mappings_result = mappings
            return generator(profile, schema, mappings_result; kwargs...)
        else
            throw(ArgumentError("llm_generate_with expects a tuple (profile, mappings) as input"))
        end
    end
end