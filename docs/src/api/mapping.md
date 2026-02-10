# Mapping Inference

Advanced column mapping inference using fuzzy string matching, value pattern analysis, and optional LLM enhancement.

## Unified API

```@docs
SDMXerWizard.infer_mappings
```

## Types

```@docs
SDMXerWizard.AdvancedMappingResult
SDMXerWizard.MappingCandidate
SDMXerWizard.MappingConfidence
SDMXerWizard.InferenceEngine
```

## Engine Creation

```@docs
SDMXerWizard.create_inference_engine
```

## Inference Functions

```@docs
SDMXerWizard.infer_column_mappings
SDMXerWizard.infer_sdmx_column_mappings
SDMXerWizard.infer_advanced_mappings
```

## Quality & Analysis

```@docs
SDMXerWizard.validate_mapping_quality
SDMXerWizard.fuzzy_match_score
SDMXerWizard.analyze_value_patterns
```

## Advanced

```@docs
SDMXerWizard.suggest_value_transformations
SDMXerWizard.analyze_mapping_coverage
SDMXerWizard.learn_from_feedback
SDMXerWizard.detect_hierarchical_relationships
```
