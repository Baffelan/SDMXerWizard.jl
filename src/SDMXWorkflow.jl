"""
SDMX Workflow System for SDMXer.jl

This module provides an integrated workflow system that orchestrates the complete
SDMX data transformation process from source analysis to script generation and validation.

Features:
- Interactive workflow management with progress tracking
- Intelligent decision making based on data characteristics
- Configurable pipeline with custom steps and parameters
- Comprehensive logging and error handling
- Support for batch processing and automation
- Integration with all SDMXer.jl components

The workflow follows these main phases:
1. Source Data Analysis & Profiling
2. Target Schema Analysis & Understanding
3. Intelligent Mapping & Inference
4. LLM-Assisted Script Generation
5. Validation & Quality Assessment
6. Output Generation & Reporting
"""

# Dependencies loaded at package level


"""
    WorkflowStep

Represents a single step in the SDMX transformation workflow with execution tracking.

This mutable struct tracks the execution state of individual workflow steps,
including timing, status, results, and error handling for comprehensive
workflow monitoring and debugging.

# Fields
- `step_id::String`: Unique identifier for this workflow step
- `step_name::String`: Human-readable name of the step
- `description::String`: Detailed description of what this step does
- `status::String`: Current status ("pending", "running", "completed", "failed", "skipped")
- `start_time::Union{DateTime, Nothing}`: When step execution started
- `end_time::Union{DateTime, Nothing}`: When step execution finished
- `duration_ms::Float64`: Execution duration in milliseconds
- `input_data::Any`: Input data for this step
- `output_data::Any`: Output data produced by this step
- `error_message::String`: Error message if step failed
- `warnings::Vector{String}`: Warning messages generated during execution
- `metrics::Dict{String, Any}`: Performance and quality metrics
- `auto_retry::Bool`: Whether to automatically retry on failure
- `max_retries::Int`: Maximum number of retry attempts
- `retry_count::Int`: Current retry attempt count

# Examples
```julia
step = WorkflowStep(
    "profile_source", "Data Profiling", "Analyze source data structure",
    "pending", nothing, nothing, 0.0, source_data, nothing, "",
    String[], Dict{String,Any}(), true, 3, 0
)

# Update step status
step.status = "running"
step.start_time = now()
```

# See also
[`SDMXWorkflow`](@ref), [`WorkflowConfig`](@ref), [`execute_workflow`](@ref)
"""
mutable struct WorkflowStep
    step_id::String
    step_name::String
    description::String
    status::String  # "pending", "running", "completed", "failed", "skipped"
    start_time::Union{DateTime, Nothing}
    end_time::Union{DateTime, Nothing}
    duration_ms::Float64
    input_data::Any
    output_data::Any
    error_message::String
    warnings::Vector{String}
    metrics::Dict{String, Any}
    auto_retry::Bool
    max_retries::Int
    retry_count::Int
end

"""
    WorkflowConfig

Configuration settings for the SDMX workflow execution.
"""
struct WorkflowConfig
    # Data source settings
    source_file_path::String
    target_dataflow_url::String
    output_directory::String

    # LLM settings
    llm_provider::Symbol  # :ollama, :openai, etc.
    llm_model::String
    enable_llm_generation::Bool
    llm_generation_timeout_s::Int

    # Processing settings
    enable_validation::Bool
    strict_validation::Bool
    auto_fix_issues::Bool
    performance_mode::Bool

    # Output settings
    generate_script::Bool
    generate_validation_report::Bool
    save_intermediate_results::Bool
    output_formats::Vector{String}  # ["script", "report", "json", "csv"]

    # Advanced settings
    custom_mapping_rules::Dict{String, String}
    custom_templates::Dict{String, Any}
    enable_interactive_mode::Bool
    log_level::String  # "debug", "info", "warn", "error"

    # Performance settings
    chunk_size::Int
    parallel_processing::Bool
    memory_limit_mb::Int
end

"""
    WorkflowResult

Complete results from workflow execution including all outputs and metadata.
"""
struct WorkflowResult
    workflow_id::String
    execution_timestamp::String
    config::WorkflowConfig
    steps::Vector{WorkflowStep}
    total_duration_ms::Float64
    overall_status::String  # "success", "partial_success", "failed"

    # Primary outputs
    source_profile::Union{SourceDataProfile, Nothing}
    target_schema::Union{DataflowSchema, Nothing}
    mapping_result::Union{AdvancedMappingResult, Nothing}
    generated_script::Union{GeneratedScript, Nothing}
    validation_result::Union{ValidationResult, Nothing}

    # Metadata and metrics
    performance_metrics::Dict{String, Any}
    quality_assessment::Dict{String, Any}
    recommendations::Vector{String}
    warnings::Vector{String}
    errors::Vector{String}

    # Output files
    output_files::Dict{String, String}
    log_file::String
end

"""
    SDMXWorkflow

Main workflow orchestrator that manages the complete SDMX transformation process.
"""
mutable struct SDMXWorkflow
    workflow_id::String
    config::WorkflowConfig
    steps::Vector{WorkflowStep}
    current_step_index::Int
    status::String

    # Component instances
    inference_engine::Union{InferenceEngine, Nothing}
    script_generator::Union{ScriptGenerator, Nothing}
    validator::Union{SDMXValidator, Nothing}

    # State tracking
    execution_start_time::Union{DateTime, Nothing}
    intermediate_data::Dict{String, Any}
    error_log::Vector{String}
    progress_callback::Union{Function, Nothing}
end

"""
    create_workflow(config::WorkflowConfig) -> SDMXWorkflow

Creates a new SDMX workflow with the specified configuration.
"""
function create_workflow(config::WorkflowConfig)
    workflow_id = "sdmx_workflow_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    # Initialize workflow steps
    steps = create_default_workflow_steps()

    workflow = SDMXWorkflow(
        workflow_id,
        config,
        steps,
        1,
        "initialized",
        nothing,
        nothing,
        nothing,
        nothing,
        Dict{String, Any}(),
        String[],
        nothing
    )

    return workflow
end

"""
    create_default_workflow_steps() -> Vector{WorkflowStep}

Creates the default workflow steps for SDMX data transformation.
"""
function create_default_workflow_steps()
    steps = Vector{WorkflowStep}()

    # Step 1: Source Data Analysis
    push!(steps, WorkflowStep(
        "source_analysis",
        "Source Data Analysis",
        "Analyze and profile the source dataset",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        true, 2, 0
    ))

    # Step 2: Target Schema Analysis
    push!(steps, WorkflowStep(
        "schema_analysis",
        "Target Schema Analysis",
        "Extract and analyze the target SDMX dataflow schema",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        true, 3, 0
    ))

    # Step 3: Intelligent Mapping
    push!(steps, WorkflowStep(
        "intelligent_mapping",
        "Intelligent Column Mapping",
        "Perform advanced mapping inference between source and target",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        true, 2, 0
    ))

    # Step 4: Script Generation
    push!(steps, WorkflowStep(
        "script_generation",
        "LLM-Assisted Script Generation",
        "Generate Tidier.jl transformation script using LLM",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        true, 1, 0
    ))

    # Step 5: Validation
    push!(steps, WorkflowStep(
        "validation",
        "SDMX-CSV Validation",
        "Validate the transformation and output quality",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        false, 0, 0
    ))

    # Step 6: Output Generation
    push!(steps, WorkflowStep(
        "output_generation",
        "Output Generation & Reporting",
        "Generate final outputs, reports, and documentation",
        "pending",
        nothing, nothing, 0.0,
        nothing, nothing, "",
        String[], Dict{String, Any}(),
        false, 0, 0
    ))

    return steps
end

"""
    execute_workflow(workflow::SDMXWorkflow) -> WorkflowResult

Executes the complete SDMX transformation workflow.
"""
function execute_workflow(workflow::SDMXWorkflow)
    workflow.execution_start_time = now()
    workflow.status = "running"

    log_info(workflow, "Starting SDMX workflow execution: $(workflow.workflow_id)")

    try
        # Execute each workflow step
        for (i, step) in enumerate(workflow.steps)
            workflow.current_step_index = i
            execute_workflow_step!(workflow, step)

            # Check if step failed and handle accordingly
            if step.status == "failed" && !step.auto_retry
                workflow.status = "failed"
                log_error(workflow, "Workflow failed at step: $(step.step_name)")
                break
            end

            # Report progress if callback is provided
            if workflow.progress_callback !== nothing
                workflow.progress_callback(i, length(workflow.steps),
                                          step.step_name, step.status)
            end
        end

        # Determine overall status
        if workflow.status != "failed"
            failed_critical_steps = sum(s.status == "failed"
                                        for s in workflow.steps if !s.auto_retry)
            if failed_critical_steps == 0
                workflow.status = "success"
            else
                workflow.status = "partial_success"
            end
        end

    catch e
        workflow.status = "failed"
        error_msg = "Workflow execution failed: $e"
        log_error(workflow, error_msg)

        # Update current step as failed
        if workflow.current_step_index <= length(workflow.steps)
            current_step = workflow.steps[workflow.current_step_index]
            current_step.status = "failed"
            current_step.error_message = error_msg
            current_step.end_time = now()
        end
    end

    # Generate workflow result
    result = generate_workflow_result(workflow)

    log_info(workflow, "Workflow completed with status: $(workflow.status)")

    return result
end

"""
    execute_workflow_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes a single workflow step with retry logic and error handling.
"""
function execute_workflow_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    log_info(workflow, "Starting step: $(step.step_name)")

    step.start_time = now()
    step.status = "running"

    try
        # Execute the specific step
        if step.step_id == "source_analysis"
            execute_source_analysis_step!(workflow, step)
        elseif step.step_id == "schema_analysis"
            execute_schema_analysis_step!(workflow, step)
        elseif step.step_id == "intelligent_mapping"
            execute_intelligent_mapping_step!(workflow, step)
        elseif step.step_id == "script_generation"
            execute_script_generation_step!(workflow, step)
        elseif step.step_id == "validation"
            execute_validation_step!(workflow, step)
        elseif step.step_id == "output_generation"
            execute_output_generation_step!(workflow, step)
        else
            error("Unknown workflow step: $(step.step_id)")
        end

        step.status = "completed"
        step.end_time = now()
        step.duration_ms = (step.end_time - step.start_time).value

        duration_s = round(step.duration_ms/1000, digits = 2)
        log_info(workflow, "Completed step: " * step.step_name * " (" * string(duration_s) * "s)")

    catch e
        step.status = "failed"
        step.error_message = string(e)
        step.end_time = now()
        step.duration_ms = (step.end_time - step.start_time).value

        log_error(workflow, "Step failed: $(step.step_name) - $e")

        # Retry logic
        if step.auto_retry && step.retry_count < step.max_retries
            step.retry_count += 1
            log_info(workflow, "Retrying step: " * step.step_name *
                     " (attempt " * string(step.retry_count) * ")")

            # Wait before retry
            sleep(2^step.retry_count)  # Exponential backoff

            execute_workflow_step!(workflow, step)
        end
    end
end

"""
    execute_source_analysis_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the source data analysis step.
"""
function execute_source_analysis_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    config = workflow.config

    # Read and profile source data
    source_data = read_source_data(config.source_file_path)
    source_profile = profile_source_data(source_data, config.source_file_path)

    # Store results
    workflow.intermediate_data["source_data"] = source_data
    workflow.intermediate_data["source_profile"] = source_profile
    step.output_data = source_profile

    # Add metrics
    step.metrics["row_count"] = source_profile.row_count
    step.metrics["column_count"] = source_profile.column_count
    step.metrics["data_quality_score"] = source_profile.data_quality_score
    step.metrics["file_size_mb"] = stat(config.source_file_path).size / (1024^2)

    # Quality warnings
    if source_profile.data_quality_score < 0.8
        quality_pct = round(source_profile.data_quality_score*100, digits = 1)
        push!(step.warnings, "Source data quality is below recommended threshold (" *
              string(quality_pct) * "%)")
    end

    if source_profile.row_count > 100000 && !config.performance_mode
        push!(step.warnings, "Large dataset detected - consider enabling performance mode")
    end
end

"""
    execute_schema_analysis_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the target schema analysis step.
"""
function execute_schema_analysis_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    config = workflow.config

    # Extract SDMX schema
    target_schema = extract_dataflow_schema(config.target_dataflow_url)

    # Store results
    workflow.intermediate_data["target_schema"] = target_schema
    step.output_data = target_schema

    # Add metrics
    required_cols = get_required_columns(target_schema)
    optional_cols = get_optional_columns(target_schema)
    codelist_cols = get_codelist_columns(target_schema)

    step.metrics["required_columns"] = length(required_cols)
    step.metrics["optional_columns"] = length(optional_cols)
    step.metrics["codelist_columns"] = length(codelist_cols)
    step.metrics["total_dimensions"] = nrow(target_schema.dimensions)
    step.metrics["total_attributes"] = nrow(target_schema.attributes)

    # Complexity warnings
    if length(required_cols) > 15
        push!(step.warnings, "Complex target schema with $(length(required_cols)) required columns")
    end
end

"""
    execute_intelligent_mapping_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the intelligent mapping step.
"""
function execute_intelligent_mapping_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    source_profile = workflow.intermediate_data["source_profile"]
    target_schema = workflow.intermediate_data["target_schema"]

    # Create inference engine if not exists
    if workflow.inference_engine === nothing
        workflow.inference_engine = create_inference_engine()
    end

    # Perform advanced mapping
    source_data = workflow.intermediate_data["source_data"]
    mapping_result = infer_advanced_mappings(workflow.inference_engine, source_profile, target_schema, source_data)

    # Store results
    workflow.intermediate_data["mapping_result"] = mapping_result
    step.output_data = mapping_result

    # Add metrics
    step.metrics["quality_score"] = mapping_result.quality_score
    step.metrics["coverage_score"] = mapping_result.coverage_analysis["required_coverage"]
    step.metrics["transformation_complexity"] = mapping_result.transformation_complexity
    step.metrics["total_mappings"] = length(mapping_result.mappings)
    step.metrics["high_confidence_mappings"] = sum(m.confidence_level == HIGH for m in mapping_result.mappings)

    # Quality warnings
    if mapping_result.quality_score < 0.7
        push!(step.warnings, "Low mapping quality score - manual review recommended")
    end

    if mapping_result.coverage_analysis["required_coverage"] < 0.8
        push!(step.warnings, "Incomplete coverage of required columns")
    end
end

"""
    execute_script_generation_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the LLM-assisted script generation step.
"""
function execute_script_generation_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    if !workflow.config.enable_llm_generation
        step.status = "skipped"
        return
    end

    source_profile = workflow.intermediate_data["source_profile"]
    target_schema = workflow.intermediate_data["target_schema"]
    mapping_result = workflow.intermediate_data["mapping_result"]

    # Create script generator if not exists
    if workflow.script_generator === nothing
        workflow.script_generator = create_script_generator(workflow.config.llm_config)
    end

    # Generate transformation script
    excel_analysis = get(workflow.intermediate_data, "excel_analysis", nothing)
    generated_script = generate_transformation_script(
        workflow.script_generator,
        source_profile,
        target_schema,
        mapping_result,
        excel_analysis
    )

    # Store results
    workflow.intermediate_data["generated_script"] = generated_script
    step.output_data = generated_script

    # Add metrics
    step.metrics["script_complexity"] = generated_script.estimated_complexity
    step.metrics["transformation_steps"] = length(generated_script.transformation_steps)
    step.metrics["validation_notes"] = length(generated_script.validation_notes)
    step.metrics["script_length_lines"] = length(split(generated_script.generated_code, '\n'))

    # Validation warnings
    if generated_script.estimated_complexity > 0.8
        push!(step.warnings, "Generated script has high complexity - thorough testing recommended")
    end

    if length(generated_script.validation_notes) > 3
        push!(step.warnings, "Multiple validation concerns in generated script")
    end
end

"""
    execute_validation_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the SDMX-CSV validation step.
"""
function execute_validation_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    if !workflow.config.enable_validation
        step.status = "skipped"
        return
    end

    target_schema = workflow.intermediate_data["target_schema"]
    source_data = workflow.intermediate_data["source_data"]

    # Create validator if not exists
    if workflow.validator === nothing
        workflow.validator = create_validator(
            target_schema;
            strict_mode = workflow.config.strict_validation,
            performance_mode = workflow.config.performance_mode
        )
    end

    # Perform validation
    validation_result = validate_sdmx_csv(workflow.validator, source_data, "workflow_dataset")

    # Store results
    workflow.intermediate_data["validation_result"] = validation_result
    step.output_data = validation_result

    # Add metrics
    step.metrics["overall_score"] = validation_result.overall_score
    step.metrics["compliance_status"] = validation_result.compliance_status
    step.metrics["total_issues"] = length(validation_result.issues)
    step.metrics["critical_issues"] = sum(issue.severity == CRITICAL for issue in validation_result.issues)
    step.metrics["error_issues"] = sum(issue.severity == ERROR for issue in validation_result.issues)

    # Critical warnings
    critical_count = step.metrics["critical_issues"]
    if critical_count > 0
        push!(step.warnings, "$critical_count critical validation issues must be addressed")
    end

    if validation_result.overall_score < 0.6
        push!(step.warnings, "Low validation score - significant data quality issues detected")
    end
end

"""
    execute_output_generation_step!(workflow::SDMXWorkflow, step::WorkflowStep)

Executes the output generation and reporting step.
"""
function execute_output_generation_step!(workflow::SDMXWorkflow, step::WorkflowStep)
    config = workflow.config
    output_files = Dict{String, String}()

    # Generate script file
    if config.generate_script && haskey(workflow.intermediate_data, "generated_script")
        generated_script = workflow.intermediate_data["generated_script"]
        script_path = joinpath(config.output_directory, "$(generated_script.script_name).jl")

        mkpath(dirname(script_path))
        open(script_path, "w") do f
            write(f, generated_script.generated_code)
        end

        output_files["script"] = script_path
    end

    # Generate validation report
    if config.generate_validation_report && haskey(workflow.intermediate_data, "validation_result")
        validation_result = workflow.intermediate_data["validation_result"]
        report_path = joinpath(config.output_directory, "validation_report.txt")

        mkpath(dirname(report_path))
        open(report_path, "w") do f
            write(f, generate_validation_report(validation_result))
        end

        output_files["validation_report"] = report_path
    end

    # Generate JSON summary
    if "json" in config.output_formats
        json_path = joinpath(config.output_directory, "workflow_summary.json")
        summary = create_workflow_summary(workflow)

        mkpath(dirname(json_path))
        open(json_path, "w") do f
            JSON3.pretty(f, summary)
        end

        output_files["json_summary"] = json_path
    end

    # Store output file paths
    workflow.intermediate_data["output_files"] = output_files
    step.output_data = output_files

    # Add metrics
    step.metrics["files_generated"] = length(output_files)
    step.metrics["total_output_size_mb"] = sum(stat(path).size for path in values(output_files)) / (1024^2)
end

"""
    generate_workflow_result(workflow::SDMXWorkflow) -> WorkflowResult

Generates the final workflow result with all outputs and metadata.
"""
function generate_workflow_result(workflow::SDMXWorkflow)
    total_duration = if workflow.execution_start_time !== nothing
        (now() - workflow.execution_start_time).value
    else
        0.0
    end

    # Extract primary outputs
    source_profile = get(workflow.intermediate_data, "source_profile", nothing)
    target_schema = get(workflow.intermediate_data, "target_schema", nothing)
    mapping_result = get(workflow.intermediate_data, "mapping_result", nothing)
    generated_script = get(workflow.intermediate_data, "generated_script", nothing)
    validation_result = get(workflow.intermediate_data, "validation_result", nothing)

    # Collect performance metrics
    performance_metrics = Dict{String, Any}(
        "total_duration_ms" => total_duration,
        "step_durations" => [s.duration_ms for s in workflow.steps],
        "successful_steps" => sum(s.status == "completed" for s in workflow.steps),
        "failed_steps" => sum(s.status == "failed" for s in workflow.steps),
        "skipped_steps" => sum(s.status == "skipped" for s in workflow.steps)
    )

    # Quality assessment
    quality_assessment = create_quality_assessment(workflow)

    # Recommendations
    recommendations = create_workflow_recommendations(workflow)

    # Collect warnings and errors
    warnings = vcat([step.warnings for step in workflow.steps]...)
    errors = workflow.error_log

    # Output files
    output_files = get(workflow.intermediate_data, "output_files", Dict{String, String}())

    return WorkflowResult(
        workflow.workflow_id,
        string(now()),
        workflow.config,
        workflow.steps,
        total_duration,
        workflow.status,
        source_profile,
        target_schema,
        mapping_result,
        generated_script,
        validation_result,
        performance_metrics,
        quality_assessment,
        recommendations,
        warnings,
        errors,
        output_files,
        ""  # log_file would be set if logging to file
    )
end

"""
    create_quality_assessment(workflow::SDMXWorkflow) -> Dict{String, Any}

Creates a comprehensive quality assessment of the workflow execution.
"""
function create_quality_assessment(workflow::SDMXWorkflow)
    assessment = Dict{String, Any}()

    # Source data quality
    if haskey(workflow.intermediate_data, "source_profile")
        source_profile = workflow.intermediate_data["source_profile"]
        assessment["source_data_quality"] = source_profile.data_quality_score
        assessment["source_completeness"] = 1.0 - (length(source_profile.missing_value_columns) / source_profile.column_count)
    end

    # Mapping quality
    if haskey(workflow.intermediate_data, "mapping_result")
        mapping_result = workflow.intermediate_data["mapping_result"]
        assessment["mapping_quality"] = mapping_result.quality_score
        assessment["mapping_coverage"] = mapping_result.coverage_analysis["required_coverage"]
        assessment["mapping_confidence"] = mean([
            m.confidence_level == HIGH ? 1.0 :
            m.confidence_level == MEDIUM ? 0.6 : 0.3
            for m in mapping_result.mappings
        ])
    end

    # Script generation quality
    if haskey(workflow.intermediate_data, "generated_script")
        generated_script = workflow.intermediate_data["generated_script"]
        assessment["script_quality"] = 1.0 - generated_script.estimated_complexity
        assessment["script_completeness"] = length(generated_script.validation_notes) == 0 ? 1.0 : 0.7
    end

    # Validation quality
    if haskey(workflow.intermediate_data, "validation_result")
        validation_result = workflow.intermediate_data["validation_result"]
        assessment["validation_score"] = validation_result.overall_score
        assessment["compliance_level"] = validation_result.compliance_status == "compliant" ? 1.0 :
                                       validation_result.compliance_status == "minor_issues" ? 0.8 :
                                       validation_result.compliance_status == "major_issues" ? 0.5 : 0.2
    end

    # Overall quality score
    quality_scores = filter(!isnan, [get(assessment, key, NaN) for key in keys(assessment) if endswith(key, "_quality") || endswith(key, "_score")])
    assessment["overall_quality"] = isempty(quality_scores) ? 0.0 : mean(quality_scores)

    return assessment
end

"""
    create_workflow_recommendations(workflow::SDMXWorkflow) -> Vector{String}

Creates actionable recommendations based on workflow execution results.
"""
function create_workflow_recommendations(workflow::SDMXWorkflow)
    recommendations = String[]

    # Failed steps
    failed_steps = filter(s -> s.status == "failed", workflow.steps)
    if !isempty(failed_steps)
        separator = ", "
        step_names = [s.step_name for s in failed_steps]
        push!(recommendations, "Address failures in: $(join(step_names, separator))")
    end

    # Data quality recommendations
    if haskey(workflow.intermediate_data, "source_profile")
        source_profile = workflow.intermediate_data["source_profile"]
        if source_profile.data_quality_score < 0.8
            push!(recommendations, "Improve source data quality before proceeding to production")
        end
    end

    # Mapping recommendations
    if haskey(workflow.intermediate_data, "mapping_result")
        mapping_result = workflow.intermediate_data["mapping_result"]
        if mapping_result.quality_score < 0.7
            push!(recommendations, "Review and refine column mappings manually")
        end
        if mapping_result.coverage_analysis["required_coverage"] < 0.9
            push!(recommendations, "Address unmapped required columns")
        end
    end

    # Script recommendations
    if haskey(workflow.intermediate_data, "generated_script")
        generated_script = workflow.intermediate_data["generated_script"]
        if generated_script.estimated_complexity > 0.7
            push!(recommendations, "Test generated script thoroughly due to high complexity")
        end
        if !isempty(generated_script.validation_notes)
            push!(recommendations, "Address validation concerns in generated script")
        end
    end

    # Performance recommendations
    total_duration_s = workflow.steps[end].duration_ms / 1000
    if total_duration_s > 300  # 5 minutes
        push!(recommendations, "Consider enabling performance mode for large datasets")
    end

    return recommendations
end

"""
    create_workflow_summary(workflow::SDMXWorkflow) -> Dict{String, Any}

Creates a JSON-serializable summary of the workflow execution.
"""
function create_workflow_summary(workflow::SDMXWorkflow)
    return Dict{String, Any}(
        "workflow_id" => workflow.workflow_id,
        "status" => workflow.status,
        "execution_time" => string(workflow.execution_start_time),
        "steps" => [
            Dict{String, Any}(
                "step_id" => step.step_id,
                "step_name" => step.step_name,
                "status" => step.status,
                "duration_ms" => step.duration_ms,
                "metrics" => step.metrics,
                "warnings" => step.warnings,
                "error" => step.error_message
            )
            for step in workflow.steps
        ],
        "config" => Dict{String, Any}(
            "source_file" => workflow.config.source_file_path,
            "target_dataflow" => workflow.config.target_dataflow_url,
            "output_directory" => workflow.config.output_directory,
            "enable_llm_generation" => workflow.config.enable_llm_generation,
            "enable_validation" => workflow.config.enable_validation
        ),
        "quality_assessment" => create_quality_assessment(workflow),
        "recommendations" => create_workflow_recommendations(workflow)
    )
end

# === LOGGING UTILITIES ===

"""
    log_info(workflow::SDMXWorkflow, message::String)

Logs an info message for the workflow.
"""
function log_info(workflow::SDMXWorkflow, message::String)
    if workflow.config.log_level in ["debug", "info"]
        println("[$(now())] INFO: $message")
    end
end

"""
    log_error(workflow::SDMXWorkflow, message::String)

Logs an error message for the workflow.
"""
function log_error(workflow::SDMXWorkflow, message::String)
    if workflow.config.log_level in ["debug", "info", "warn", "error"]
        println("[$(now())] ERROR: $message")
    end
    push!(workflow.error_log, message)
end

"""
    generate_workflow_report(result::WorkflowResult; format::String = "text") -> String

Generates a comprehensive workflow execution report.
"""
function generate_workflow_report(result::WorkflowResult; format::String = "text")
    if format == "text"
        return generate_text_workflow_report(result)
    elseif format == "html"
        return generate_html_workflow_report(result)
    else
        error("Unsupported report format: $format")
    end
end

"""
    generate_text_workflow_report(result::WorkflowResult) -> String

Generates a text-based workflow report.
"""
function generate_text_workflow_report(result::WorkflowResult)
    report = """
═══════════════════════════════════════════════════════════════
                    SDMX WORKFLOW EXECUTION REPORT
═══════════════════════════════════════════════════════════════

Workflow ID: $(result.workflow_id)
Execution Date: $(result.execution_timestamp)
Overall Status: $(uppercase(result.overall_status))
Total Duration: $(round(result.total_duration_ms/1000, digits=2)) seconds

═══ CONFIGURATION ═══
Source File: $(result.config.source_file_path)
Target Dataflow: $(result.config.target_dataflow_url)
Output Directory: $(result.config.output_directory)
LLM Generation: $(result.config.enable_llm_generation ? "Enabled" : "Disabled")
Validation: $(result.config.enable_validation ? "Enabled" : "Disabled")

═══ WORKFLOW STEPS ═══
"""

    for (i, step) in enumerate(result.steps)
        status_symbol = step.status == "completed" ? "✅" :
                       step.status == "failed" ? "❌" :
                       step.status == "skipped" ? "⏸️" : "⏳"

        report *= "$i. $status_symbol $(step.step_name) ($(step.status))\n"
        report *= "   Duration: $(round(step.duration_ms/1000, digits=2))s\n"

        if !isempty(step.warnings)
            separator = "; "
            report *= "   Warnings: $(join(step.warnings, separator))\n"
        end

        if !isempty(step.error_message)
            report *= "   Error: $(step.error_message)\n"
        end

        if !isempty(step.metrics)
            key_metrics = ["quality_score", "coverage_score", "complexity", "row_count", "column_count"]
            for metric in key_metrics
                if haskey(step.metrics, metric)
                    report *= "   $(metric): $(step.metrics[metric])\n"
                end
            end
        end

        report *= "\n"
    end

    # Quality Assessment
    if !isempty(result.quality_assessment)
        report *= "═══ QUALITY ASSESSMENT ═══\n"
        for (metric, score) in result.quality_assessment
            if isa(score, Number)
                report *= "$(metric): $(round(score * 100, digits=1))%\n"
            else
                report *= "$(metric): $score\n"
            end
        end
        report *= "\n"
    end

    # Recommendations
    if !isempty(result.recommendations)
        report *= "═══ RECOMMENDATIONS ═══\n"
        for (i, rec) in enumerate(result.recommendations)
            report *= "$i. $rec\n"
        end
        report *= "\n"
    end

    # Output Files
    if !isempty(result.output_files)
        report *= "═══ OUTPUT FILES ═══\n"
        for (type, path) in result.output_files
            report *= "$(type): $path\n"
        end
        report *= "\n"
    end

    report *= "═══════════════════════════════════════════════════════════════\n"

    return report
end
