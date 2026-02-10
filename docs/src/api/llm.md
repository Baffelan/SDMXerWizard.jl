# LLM Integration

Core functions for setting up and interfacing with LLM providers via PromptingTools.jl.

## Provider Setup

```@docs
SDMXerWizard.setup_sdmx_llm
SDMXerWizard.llm_provider
SDMXerWizard.script_style_enum
```

## LLM Calls

```@docs
SDMXerWizard.sdmx_aigenerate
SDMXerWizard.sdmx_aiextract
```

## Types & Constants

```@docs
SDMXerWizard.LLMProvider
SDMXerWizard.ScriptStyle
SDMXerWizard.LLM_PROVIDERS
```

## Data Sources

```@docs
SDMXerWizard.DataSource
SDMXerWizard.FileSource
SDMXerWizard.NetworkSource
SDMXerWizard.MemorySource
SDMXerWizard.CSVSource
SDMXerWizard.ExcelSource
SDMXerWizard.URLSource
SDMXerWizard.DataFrameSource
SDMXerWizard.read_data
SDMXerWizard.source_info
SDMXerWizard.validate_source
SDMXerWizard.data_source
SDMXerWizard.read_source_data
```

## Anonymization

```@docs
SDMXerWizard.AnonymizationConfig
SDMXerWizard.anonymize_source_data
SDMXerWizard.anonymize_column_values
SDMXerWizard.summarize_anonymized_data
```

## Context & Metadata

```@docs
SDMXerWizard.extract_structural_context
SDMXerWizard.extract_data_source_context
SDMXerWizard.build_transformation_context
SDMXerWizard.generate_enhanced_transformation_script
SDMXerWizard.prepare_data_preview
SDMXerWizard.prepare_schema_context
SDMXerWizard.prepare_source_structure
```

## Excel Analysis

```@docs
SDMXerWizard.analyze_excel_with_ai
```

## Result Types

```@docs
SDMXerWizard.SDMXMappingResult
SDMXerWizard.SDMXTransformationScript
SDMXerWizard.ExcelStructureAnalysis
SDMXerWizard.SDMXStructuralContext
SDMXerWizard.DataSourceContext
SDMXerWizard.TransformationContext
```
