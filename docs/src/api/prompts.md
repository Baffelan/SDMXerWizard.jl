# Prompts

Prompt construction functions for building structured prompts for LLM use.

## Cross-Dataflow Join Prompts

```@docs
SDMXerWizard.create_join_analysis_prompt
SDMXerWizard.create_indicator_classification_prompt
SDMXerWizard.create_unit_inference_prompt
```

## Prompt Templates

```@docs
SDMXerWizard.PromptTemplate
SDMXerWizard.TransformationScenario
SDMXerWizard.ComprehensivePrompt
```

## Prompt Generation

```@docs
SDMXerWizard.create_prompt_template
SDMXerWizard.build_comprehensive_prompt
SDMXerWizard.generate_transformation_prompt
```

## Template Builders

```@docs
SDMXerWizard.create_mapping_template
SDMXerWizard.create_script_template
SDMXerWizard.create_excel_analysis_template
```

## Context Sections

```@docs
SDMXerWizard.build_sdmx_context_section
SDMXerWizard.build_source_analysis_section
SDMXerWizard.build_code_mapping_section
```

## Template & Scenario Constants

Pre-built `ScriptTemplate` instances:
- `TIDIER_TEMPLATE` -- Tidier.jl transformation template
- `DATAFRAMES_TEMPLATE` -- DataFrames.jl transformation template
- `MIXED_TEMPLATE` -- mixed-style transformation template

Pre-built `TransformationScenario` instances:
- `EXCEL_SCENARIO` -- Excel-to-SDMX transformation scenario
- `CSV_SCENARIO` -- CSV-to-SDMX transformation scenario
- `PIVOTING_SCENARIO` -- pivot/reshape transformation scenario
- `CODE_MAPPING_SCENARIO` -- codelist mapping transformation scenario

## Provider & Style Constants

`LLMProvider` enum instances: `OPENAI`, `ANTHROPIC`, `OLLAMA`, `MISTRAL`, `AZURE_OPENAI`, `GROQ`, `TOGETHER`, `FIREWORKS`, `DATABRICKS`, `GOOGLE`.

`ScriptStyle` enum instances: `DATAFRAMES`, `TIDIER`, `MIXED`.
