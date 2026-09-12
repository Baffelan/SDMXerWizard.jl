# Data Sources and Anonymisation

Reading CSV and Excel files into DataFrames, and replacing their values with
synthetic ones before a model sees them.

## Source types

```@docs
SDMXerWizard.DataSource
SDMXerWizard.FileSource
SDMXerWizard.NetworkSource
SDMXerWizard.MemorySource
SDMXerWizard.CSVSource
SDMXerWizard.ExcelSource
SDMXerWizard.URLSource
SDMXerWizard.DataFrameSource
```

## Reading

```@docs
SDMXerWizard.read_source_data
SDMXerWizard.read_data
SDMXerWizard.data_source
SDMXerWizard.source_info
SDMXerWizard.validate_source
```

## Anonymisation

```@docs
SDMXerWizard.AnonymizationConfig
SDMXerWizard.anonymize_source_data
SDMXerWizard.anonymize_column_values
SDMXerWizard.summarize_anonymized_data
```
