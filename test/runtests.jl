using Test
using SDMXer
using SDMXerWizard
using DataFrames
using HTTP
using PromptingTools
using Dates

@testset "SDMXerWizard.jl Tests" begin

    # Run Aqua quality checks first
    @testset "Code Quality (Aqua.jl)" begin
        include("aqua.jl")
    end

    @testset "LLM Integration" begin
        # Test LLM configuration setup
        ollama_config = setup_sdmx_llm(:ollama; model="llama2")
        @test ollama_config isa SDMXerWizard.LLMProvider

        # Test that LLM provider enums are exported
        @test SDMXerWizard.OLLAMA isa SDMXerWizard.LLMProvider
        @test SDMXerWizard.OPENAI isa SDMXerWizard.LLMProvider
        @test SDMXerWizard.ANTHROPIC isa SDMXerWizard.LLMProvider
        @test SDMXerWizard.GOOGLE isa SDMXerWizard.LLMProvider

        # Test enum string conversion
        @test length(string(SDMXerWizard.OLLAMA)) > 0
        @test length(string(SDMXerWizard.OPENAI)) > 0
    end

    @testset "Excel Structure Analysis" begin
        # Create a simple test Excel file in memory simulation
        test_excel_data = DataFrame(
            Country = ["FJ", "TV", "FJ", "TV"],
            Y2020 = [85.2, 92.1, 78.9, 89.7],
            Y2021 = [87.1, 93.5, 82.3, 91.2],
            Y2022 = [88.3, 94.1, 84.1, 92.6]
        )
        @test nrow(test_excel_data) == 4
        @test ncol(test_excel_data) == 4
    end

    @testset "Advanced Mapping Inference" begin
        # Test inference engine creation
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        @test inference_engine.fuzzy_threshold == 0.6
        @test inference_engine.min_confidence == 0.2

        # Test fuzzy matching
        fuzzy_score = SDMXerWizard.fuzzy_match_score("country", "GEO_PICT")
        @test fuzzy_score > 0.3  # Should find semantic similarity with improved algorithm
        @test fuzzy_score <= 1.0

        # Test value pattern analysis
        mock_codelist = DataFrame(code_id=["FJ", "VU", "TV"], name=["Fiji", "Vanuatu", "Tuvalu"])
        test_values = ["FJ", "VU", "TV"]
        pattern_analysis = SDMXerWizard.analyze_value_patterns(test_values, mock_codelist)
        @test haskey(pattern_analysis, "exact_matches")
        @test pattern_analysis["exact_matches"] == 3
        @test pattern_analysis["confidence_score"] == 1.0
    end

    @testset "Hierarchical Relationships" begin
        hierarchical_profile = profile_source_data(DataFrame(
            broad_cat = ["A", "A", "B", "B"],
            detail_cat = ["A1", "A2", "B1", "B2"]
        ), "test")

        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
        hierarchy_analysis = SDMXerWizard.detect_hierarchical_relationships(hierarchical_profile, schema_for_test)
        @test haskey(hierarchy_analysis, "potential_hierarchies")
        @test haskey(hierarchy_analysis, "parent_child_relationships")
    end

    @testset "Script Generation" begin
        # Test script generator setup
        test_llm_config = setup_sdmx_llm(:ollama; model="llama2")
        script_generator = create_script_generator(:ollama, "llama2")
        @test script_generator.provider == :ollama
        @test script_generator.model == "llama2"
        @test script_generator.include_validation == true
        @test script_generator.include_comments == true
        @test script_generator.tidier_style == "pipes"
        @test haskey(script_generator.templates, "standard_transformation")
        @test haskey(script_generator.templates, "pivot_transformation")
        @test haskey(script_generator.templates, "excel_multi_sheet")
        @test haskey(script_generator.templates, "simple_csv")

        # Test template creation
        standard_template = SDMXerWizard.create_standard_template()
        @test standard_template.template_name == "standard_transformation"
        @test haskey(standard_template.template_sections, "header")
        @test haskey(standard_template.template_sections, "data_loading")
        @test haskey(standard_template.template_sections, "transformations")
        @test haskey(standard_template.template_sections, "validation")
        @test haskey(standard_template.template_sections, "output")
        @test "DataFrames" in standard_template.required_packages
        @test "Tidier" in standard_template.required_packages
    end

    @testset "Transformation Steps" begin
        # Test transformation step building
        test_data = DataFrame(
            country = ["FJ", "TV", "FJ", "TV"],
            year = [2020, 2020, 2021, 2021],
            value = [85.2, 92.1, 87.1, 89.3],
            category = ["A", "A", "B", "B"]
        )

        profile = profile_source_data(test_data, "test.csv")
        @test profile.row_count == 4
        @test profile.column_count == 4

        # Get test schema
        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")

        # Create inference engine and run mapping
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test, test_data)

        transformation_steps = build_transformation_steps(advanced_mapping, profile, schema_for_test)
        @test length(transformation_steps) > 0
        @test any(step -> step.operation_type == "read", transformation_steps)
        @test any(step -> step.operation_type == "mutate" || step.operation_type == "select", transformation_steps)
        @test any(step -> step.operation_type == "write", transformation_steps)
    end

    @testset "Loading Code Generation" begin
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )

        # Test CSV loading code
        profile = profile_source_data(test_data, "test.csv")
        csv_loading_code = SDMXerWizard.get_loading_code(profile)
        @test occursin("CSV.read", csv_loading_code)
        @test occursin(profile.file_path, csv_loading_code)

        # Test Excel profile loading code
        excel_profile = SourceDataProfile(
            "test.xlsx", "xlsx", 4, 4, profile.columns, 1.0,
            String[], String[], String[]
        )
        excel_loading_code = SDMXerWizard.get_loading_code(excel_profile)
        @test occursin("XLSX.readtable", excel_loading_code)
    end

    @testset "Transformation Logic" begin
        # Test transformation logic generation
        test_mapping = SDMXerWizard.MappingCandidate(
            "country", "GEO_PICT", 1.0, SDMXerWizard.HIGH, "exact",
            Dict{String, Any}(), nothing, String[]
        )
        basic_logic = SDMXerWizard.generate_transformation_logic(test_mapping)
        @test occursin("GEO_PICT = country", basic_logic)
    end

    @testset "Validation Logic" begin
        # Setup test data
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )
        profile = profile_source_data(test_data, "test.csv")
        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test, test_data)

        # Test validation logic creation
        validation_logic = SDMXerWizard.create_validation_logic(advanced_mapping, schema_for_test)
        @test occursin("required_cols", validation_logic)
        @test occursin("missing_cols", validation_logic)
        @test occursin("missing_count", validation_logic)
    end

    @testset "Script Complexity" begin
        # Setup test data
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )
        profile = profile_source_data(test_data, "test.csv")
        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test, test_data)
        transformation_steps = build_transformation_steps(advanced_mapping, profile, schema_for_test)

        # Test script complexity calculation
        complexity = SDMXerWizard.calculate_script_complexity(transformation_steps, advanced_mapping)
        @test complexity >= 0.0
        @test complexity <= 1.0
    end

    @testset "Template Selection" begin
        # Setup
        script_generator = create_script_generator(:ollama, "llama2")
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )
        profile = profile_source_data(test_data, "test.csv")

        # Test template selection
        selected_template = SDMXerWizard.select_template(script_generator, profile, nothing, "")
        @test selected_template.template_name in ["standard_transformation", "simple_csv"]

        # CSV template should be selected for simple CSV files
        simple_csv_profile = SourceDataProfile(
            "simple.csv", "csv", 4, 3, profile.columns[1:3], 1.0,
            String[], String[], String[]
        )
        csv_template = SDMXerWizard.select_template(script_generator, simple_csv_profile, nothing, "")
        @test csv_template.template_name == "simple_csv"
    end

    @testset "Script Validation" begin
        # Setup test data
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )
        profile = profile_source_data(test_data, "test.csv")
        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test, test_data)
        transformation_steps = build_transformation_steps(advanced_mapping, profile, schema_for_test)

        # Create mock script
        mock_script = SDMXerWizard.GeneratedScript(
            "test_script", "test.csv", "SPC:DF_BP50",
            "using DataFrames, Tidier\nsource_data = CSV.read(\"test.csv\", DataFrame)\ntransformed_data = source_data |> @mutate(GEO_PICT = country)\nCSV.write(\"output.csv\", transformed_data)",
            transformation_steps, 0.3, String[], String[], "standard_transformation", string(now())
        )

        # Test script validation
        script_validation = SDMXerWizard.validate_generated_script(mock_script)
        @test haskey(script_validation, "syntax_issues")
        @test haskey(script_validation, "missing_elements")
        @test haskey(script_validation, "recommendations")
        @test haskey(script_validation, "overall_quality")

        # Test script preview
        preview_text = SDMXerWizard.preview_script_output(mock_script; max_lines=20)
        @test length(preview_text) > 0
        @test occursin("PREVIEW", uppercase(preview_text))
    end

    @testset "Script Guidance" begin
        # Setup test data
        test_data = DataFrame(
            country = ["FJ", "TV"],
            year = [2020, 2021],
            value = [85.2, 92.1]
        )
        profile = profile_source_data(test_data, "test.csv")
        schema_for_test = extract_dataflow_schema("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all")
        inference_engine = SDMXerWizard.create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test, test_data)
        transformation_steps = build_transformation_steps(advanced_mapping, profile, schema_for_test)

        # Test guidance creation
        validation_notes, user_guidance = SDMXerWizard.create_script_guidance(advanced_mapping, transformation_steps, profile, schema_for_test)
        @test length(validation_notes) > 0
        @test length(user_guidance) > 0
        @test any(note -> occursin("required", lowercase(note)), validation_notes)
    end

end # End of main testset
