"""
    SDMXerWizard

A library that a language model uses, rather than a library that calls one.
Built on `SDMXer` (SDMX.jl), it profiles source data, infers mappings to an
SDMx dataflow, plans a transformation, and validates the result. The calling
model orchestrates the steps and writes the transformation script; this
package supplies the facts and checks the outcome.

# Layers

**Core** (pure Julia, returns structs)
    read_source_data / profile_source_data  ─→  SourceDataProfile
    infer_mappings(source, schema; method=:heuristic | :fuzzy | :advanced)
    infer_advanced_mappings  ─→  AdvancedMappingResult
    build_transformation_steps  ─→  Vector{TransformationStep}
    select_template / default_templates  ─→  ScriptTemplate
    anonymize_source_data  ─→  DataFrame with synthetic values

**Tools** (`SDMXerWizard.Tools`, returns JSON-friendly Dicts)
    Session  ─→  handles for loaded schemas and sources
    load_schema, load_source, infer_mappings,
    transformation_plan, run_script, validate_csv, compare_with_published

**MCP server**
    serve_mcp()  ─→  stdio Model Context Protocol server over a Session
    mcp_tools(session)  ─→  the same tools as MCPTool values

Dataflow discovery and code browsing are left to the SDMx MCP gateway
(https://github.com/Baffelan/sdmx-mcp-gateway), registered alongside this
server; `load_schema` accepts the endpoint and dataflow id it reports.

See also: `SDMXer` (SDMX.jl) for core SDMx parsing, validation, and joins.
"""
module SDMXerWizard

using SDMXer
using HTTP, JSON3, DataFrames, CSV, Statistics, StatsBase, Dates, EzXML, YAML, XLSX
using Logging

include("SDMXDataSources.jl")
include("SDMXDataProfiling.jl")
include("SDMXAnonymization.jl")
include("SDMXMappingInference.jl")
include("SDMXScriptGeneration.jl")
include("Tools.jl")
include("MCPServer.jl")

# === TOOL SURFACE ===
export Tools, serve_mcp, mcp_tools, mcp_prompts

# === RE-EXPORTED FROM SDMXer ===
export DataflowSchema, extract_dataflow_schema, extract_all_codelists
export get_required_columns, get_optional_columns

# === DATA SOURCES ===
export DataSource, FileSource, NetworkSource, MemorySource
export CSVSource, ExcelSource, URLSource, DataFrameSource
export read_data, source_info, validate_source, data_source, read_source_data

# === DATA PROFILING ===
export ColumnProfile, SourceDataProfile
export profile_source_data, profile_column, detect_column_type_and_patterns
export print_source_profile, suggest_column_mappings

# === DATA ANONYMIZATION ===
export AnonymizationConfig, anonymize_source_data, anonymize_column_values, summarize_anonymized_data

# === MAPPING INFERENCE ===
export MappingCandidate, MappingConfidence, AdvancedMappingResult, InferenceEngine
export infer_mappings, infer_advanced_mappings
export create_inference_engine, validate_mapping_quality
export suggest_value_transformations, analyze_mapping_coverage, learn_from_feedback
export fuzzy_match_score, analyze_value_patterns, detect_hierarchical_relationships

# === SCRIPT PLANNING ===
export ScriptTemplate, TransformationStep, GeneratedScript
export default_templates, select_template, build_transformation_steps, get_loading_code
export validate_generated_script, preview_script_output

end # module SDMXerWizard
