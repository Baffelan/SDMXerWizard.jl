"""
    SDMXerWizard

LLM-powered extension for `SDMXer` (SDMX.jl). Adds data profiling, mapping inference,
script generation, and end-to-end workflow orchestration.

# Workflow paths

**Data profiling**
    profile_source_data  ─→  SourceDataProfile  (with ColumnProfile per column)

**Mapping inference** (unified API)
    infer_mappings(source, schema; method=:heuristic)   # no LLM
    infer_mappings(source, schema; method=:fuzzy)        # fuzzy string matching
    infer_mappings(source, schema; method=:llm)          # LLM-enhanced
    infer_mappings(source, schema; method=:advanced)     # all engines combined
    ─→ AdvancedMappingResult / SDMXMappingResult

**Script generation**
    create_script_generator  ─→  ScriptGenerator
    generate_transformation_script  ─→  GeneratedScript

**Single-dataflow workflow**
    WorkflowConfig  ─→  create_workflow  ─→  SDMXWorkflow
    execute_workflow  ─→  WorkflowResult
    generate_workflow_report  ─→  formatted report string

**Cross-dataflow join workflow**
    JoinWorkflowConfig  ─→  execute_join_workflow  ─→  Dict with JoinResult

**LLM setup**
    setup_sdmx_llm(:provider; model=...)  # configure PromptingTools backend
    sdmx_aigenerate  /  sdmx_aiextract    # low-level LLM calls

**Cross-dataflow LLM helpers**
    suggest_analysis_dataflows  ─→  recommended dataflow IDs for a research question
    infer_unit_conversion       ─→  LLM-suggested unit conversion strategy
    generate_join_script        ─→  GeneratedScript for joining dataflows
    infer_indicator_semantics  /  suggest_comparable_indicators

**Prompt construction** (no LLM calls — build prompts for external use)
    create_join_analysis_prompt  /  create_indicator_classification_prompt
    create_unit_inference_prompt

# Core types

- `JoinWorkflowConfig`: configuration for multi-dataflow join workflow
- `WorkflowConfig`: configuration for single-dataflow transformation workflow
- `WorkflowResult`: outcome of `execute_workflow`
- `SourceDataProfile`: column-level profile of a source dataset
- `AdvancedMappingResult`: mapping results from `infer_mappings`
- `GeneratedScript`: transformation script produced by `generate_transformation_script`
- `ScriptGenerator`: configured script generation engine

See also: `SDMXer` (SDMX.jl) for core SDMX parsing, validation, and deterministic joins.
"""
module SDMXerWizard

# Load all dependencies at package level
using SDMXer
using PromptingTools
using HTTP, JSON3, DataFrames, CSV, Statistics, StatsBase, Dates, EzXML, YAML, XLSX

# Include all module files
include("SDMXDataSources.jl")
include("SDMXDataProfiling.jl")
include("SDMXAnonymization.jl")
include("SDMXPromptingIntegration.jl")
include("SDMXMetadataContext.jl")
include("SDMXEnhancedTransformation.jl")
include("SDMXMappingInference.jl")
include("SDMXScriptGeneration.jl")
include("SDMXWorkflow.jl")
include("SDMXLLMPipelineOps.jl")
include("SDMXPromptGeneration.jl")
include("SDMXJoinPrompts.jl")
include("SDMXCrossDataflowLLM.jl")
include("SDMXJoinWorkflow.jl")

# === CORE DATA STRUCTURES ===

# Data Source Types - Abstractions for various data input sources
export DataSource, FileSource, NetworkSource, MemorySource
export CSVSource, ExcelSource, URLSource, DataFrameSource

# LLM Provider Configuration - Enumeration types for LLM providers and output styles
export LLMProvider, ScriptStyle
export OPENAI, ANTHROPIC, OLLAMA, MISTRAL, AZURE_OPENAI, GROQ, TOGETHER, FIREWORKS, DATABRICKS, GOOGLE
export DATAFRAMES, TIDIER, MIXED

# Result & Analysis Types - Data structures for LLM analysis results and transformations
export SDMXMappingResult, SDMXTransformationScript, ExcelStructureAnalysis
export SDMXStructuralContext, DataSourceContext, TransformationContext
export MappingCandidate, MappingConfidence, AdvancedMappingResult, InferenceEngine
export ScriptTemplate, TransformationStep, GeneratedScript, ScriptGenerator
export SDMXWorkflow, WorkflowStep, WorkflowConfig, WorkflowResult
export PromptTemplate, TransformationScenario, ComprehensivePrompt

# Template & Scenario Constants - Predefined templates and scenarios for common use cases
export TIDIER_TEMPLATE, DATAFRAMES_TEMPLATE, MIXED_TEMPLATE
export EXCEL_SCENARIO, CSV_SCENARIO, PIVOTING_SCENARIO, CODE_MAPPING_SCENARIO
export LLM_PROVIDERS

# === DATA SOURCE OPERATIONS ===
# Functions for reading and validating data from various source types
export read_data, source_info, validate_source, data_source, read_source_data

# === DATA PROFILING ===
# Types and functions for comprehensive data profiling and analysis
export ColumnProfile, SourceDataProfile
export profile_source_data, profile_column, detect_column_type_and_patterns
export print_source_profile, suggest_column_mappings

# === DATA ANONYMIZATION ===
# AI-free anonymization utilities for privacy-preserving workflows
export AnonymizationConfig, anonymize_source_data, anonymize_column_values, summarize_anonymized_data

# === LLM INTEGRATION CORE ===
# Core functions for setting up and interfacing with various LLM providers
export setup_sdmx_llm, llm_provider, script_style_enum
export sdmx_aigenerate, sdmx_aiextract

# === EXCEL ANALYSIS ===
# Specialized functions for analyzing Excel file structures and data layouts
export analyze_excel_with_ai

# === MAPPING & INFERENCE ===
# Advanced mapping inference using fuzzy matching and LLM-enhanced suggestions
export infer_mappings  # New unified API
export infer_column_mappings, infer_sdmx_column_mappings, infer_advanced_mappings
export create_inference_engine, validate_mapping_quality
export suggest_value_transformations, analyze_mapping_coverage, learn_from_feedback
export fuzzy_match_score, analyze_value_patterns, detect_hierarchical_relationships

# === SCRIPT GENERATION ===
# Functions for generating Tidier.jl/DataFrames.jl transformation scripts
export generate_transformation_script, generate_transformation_script_text, create_script_generator
export build_transformation_steps, validate_generated_script, preview_script_output

# === CONTEXT & METADATA ===
# Functions for extracting and building context for LLM-enhanced transformations
export extract_structural_context, extract_data_source_context, build_transformation_context
export generate_enhanced_transformation_script
export prepare_data_preview, prepare_schema_context, prepare_source_structure

# === PROMPT GENERATION ===
# Comprehensive prompt generation system for various LLM transformation scenarios
export create_prompt_template, build_comprehensive_prompt, generate_transformation_prompt
export create_mapping_template, create_script_template, create_excel_analysis_template
export build_sdmx_context_section, build_source_analysis_section, build_code_mapping_section

# === WORKFLOW ORCHESTRATION ===
# High-level workflow management for end-to-end SDMX data transformation pipelines
export create_workflow, execute_workflow, generate_workflow_report, ai_sdmx_workflow

# === CROSS-DATAFLOW JOIN PROMPTS ===
# Prompt construction for LLM-assisted cross-dataflow analysis (no LLM calls)
export create_join_analysis_prompt, create_indicator_classification_prompt, create_unit_inference_prompt

# === CROSS-DATAFLOW LLM INFERENCE ===
# LLM-powered cross-dataflow analysis and script generation
export infer_unit_conversion, suggest_analysis_dataflows, generate_join_script
export infer_indicator_semantics, suggest_comparable_indicators

# === CROSS-DATAFLOW JOIN WORKFLOW ===
# End-to-end workflow for multi-dataflow joins
export JoinWorkflowConfig, execute_join_workflow

end # module SDMXerWizard
