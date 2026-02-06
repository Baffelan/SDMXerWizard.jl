# SDMXerWizard.jl

[![Build Status](https://github.com/Baffelan/SDMXerWizard.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Baffelan/SDMXerWizard.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/Baffelan/SDMXerWizard.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Baffelan/SDMXerWizard.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Extension package for SDMXer.jl that provides LLM-powered data transformation and mapping capabilities. Automatically analyze data structures, infer column mappings, and generate transformation scripts for SDMX (Statistical Data and Metadata eXchange) compliance.

## Requirements

- Julia 1.11 or higher
- SDMXer.jl package (automatically installed as dependency)
- See [Project.toml](Project.toml) for full dependencies

## Features

- **Multi-Provider LLM Support**: Ollama, OpenAI, Anthropic, Google, Mistral, Groq
- **Intelligent Mapping**: Advanced fuzzy matching and semantic analysis
- **Script Generation**: Automatic Tidier.jl transformation code generation
- **Workflow Orchestration**: Complete transformation pipelines
- **Excel Analysis**: Multi-sheet workbook structure understanding
- **Pattern Recognition**: Hierarchical relationship detection

## Philosophy

The package makes some (radical?) design choices. Two of these are:

- user does not need to know SDMX REST api syntax. _As much as possible_ the package works starting from the developer API query link given by the .Stat Data Explorer.
- AI and LLMs in particular are used to provide draft code, that the user can integrate, rather than answers. The usage of SDMX data in many Official Statistics or critically important activities encourage a careful usage of generative AI.

## Installation

```julia
using Pkg
# Install SDMXerWizard.jl (SDMXer.jl will be installed automatically)
Pkg.add(url="https://github.com/Baffelan/SDMXerWizard.jl")
```

## Quick Start

### Basic Usage

```julia
using SDMXerWizard

# Load SDMX schema from API
url = "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all"
schema = extract_dataflow_schema(url)

# Read and analyze source data
source_data = read_source_data("my_data.csv")
profile = profile_source_data(source_data, "my_data.csv")

# Infer column mappings
mappings = infer_mappings(source_data, schema; method=:advanced)
```

### With LLM-Enhanced Transformation

```julia
using SDMXerWizard

# Configure LLM (optional - defaults to Ollama)
setup_sdmx_llm(:ollama; model="llama3")

# Load schema and data
schema = extract_dataflow_schema(url)
source_data = read_source_data("my_data.csv")
profile = profile_source_data(source_data, "my_data.csv")

# Get detailed mapping analysis
engine = create_inference_engine(fuzzy_threshold=0.7)
mapping_result = infer_advanced_mappings(engine, profile, schema, source_data)

# Generate transformation script
generator = create_script_generator(:ollama, "llama3")
script = generate_transformation_script(
    generator, profile, schema, mapping_result,
    output_path="transformed_data.csv"
)

# Save the generated script
write("transform_to_sdmx.jl", script.generated_code)
```

## LLM Provider Configuration

### Local Models (Ollama)

```julia
# Default Ollama setup
setup_sdmx_llm(:ollama; model="llama3")

# Custom Ollama endpoint
setup_sdmx_llm(:ollama;
    model="mixtral",
    base_url="http://localhost:11434"
)
```

### Cloud Providers

#### OpenAI

```julia
# Set API key via environment variable
ENV["OPENAI_API_KEY"] = "sk-..."
setup_sdmx_llm(:openai; model="gpt-4")

# Or load from .env file
setup_sdmx_llm(:openai; model="gpt-4", env_file=".env")
```

#### Google Gemini

```julia
# IMPORTANT: Set BEFORE importing SDMXerWizard
ENV["GOOGLE_API_KEY"] = "AIza..."
using SDMXerWizard
setup_sdmx_llm(:google; model="gemini-1.5-flash")
```

#### Anthropic Claude

```julia
ENV["ANTHROPIC_API_KEY"] = "sk-ant-..."
setup_sdmx_llm(:anthropic; model="claude-3-sonnet")
```

### .env File Format

```yaml
OPENAI_API_KEY: "sk-..."
GOOGLE_API_KEY: "AIza..."
ANTHROPIC_API_KEY: "sk-ant-..."
MISTRAL_API_KEY: "..."
GROQ_API_KEY: "..."
```

## Key Capabilities

### LLM Integration

Query LLMs directly with SDMX context for data analysis and mapping suggestions. Supports multiple providers including Ollama for local inference and cloud providers for advanced models.

### Advanced Mapping Inference

Intelligent column mapping using multiple strategies:

- **Heuristic**: Rule-based matching using column names and patterns
- **Fuzzy**: String similarity matching with configurable thresholds
- **LLM**: Semantic understanding for complex mappings
- **Advanced**: Combines all methods with confidence scoring

### Script Generation

Automatically generate Tidier.jl transformation scripts with:

- Validation checks for data quality
- Custom transformations for specific columns
- Multiple output formats (CSV, Parquet, etc.)
- Comments and documentation

### Workflow Orchestration

Complete end-to-end pipelines that combine all capabilities to transform raw data into SDMX-compliant format with minimal manual intervention.

## Advanced Features

- **Hierarchical Relationship Detection**: Automatically identify parent-child relationships in data structures
- **Pattern Analysis**: Match data values against SDMX codelists with confidence scoring
- **Transformation Pipeline Builder**: Create multi-step transformation workflows
- **Excel Structure Analysis**: Understand complex multi-sheet workbooks
- **Data Quality Validation**: Check conformance to SDMX standards

## API Reference

### Core Functions

| Function                                            | Description            |
| --------------------------------------------------- | ---------------------- |
| `setup_sdmx_llm(provider; kwargs...)`               | Configure LLM provider |
| `read_source_data(file_path; kwargs...)`            | Read CSV/Excel data    |
| `profile_source_data(data, file_path)`              | Profile data structure |
| `infer_mappings(source, schema; method, kwargs...)` | Unified mapping API    |

### Advanced Functions

| Function                                                              | Description                  |
| --------------------------------------------------------------------- | ---------------------------- |
| `create_inference_engine(kwargs...)`                                  | Create mapping engine        |
| `infer_advanced_mappings(engine, profile, schema, data)`              | Run advanced inference       |
| `create_script_generator(provider, model; kwargs...)`                 | Create code generator        |
| `generate_transformation_script(generator, profile, schema, mapping)` | Generate transformation code |
| `create_workflow(source, schema, output; kwargs...)`                  | Define complete workflow     |
| `execute_workflow(workflow)`                                          | Run transformation pipeline  |

### Utility Functions

| Function                                               | Description                      |
| ------------------------------------------------------ | -------------------------------- |
| `analyze_excel_structure(filepath)`                    | Analyze Excel workbook structure |
| `detect_hierarchical_relationships(profile, schema)`   | Find data hierarchies            |
| `fuzzy_match_score(str1, str2)`                        | Calculate string similarity      |
| `validate_generated_script(script)`                    | Validate script quality          |
| `build_transformation_steps(mapping, profile, schema)` | Build transformation steps       |

## Transformation Templates

The package includes pre-built templates that are automatically selected based on data complexity:

- **Standard**: Basic column mapping and renaming
- **Pivot**: Wide to long format conversion
- **Excel Multi-Sheet**: Complex workbook handling
- **Simple CSV**: Optimized for simple CSV files

## Testing

Run the test suite:

```julia
using Pkg
Pkg.test("SDMXerWizard")
```

All 72 tests should pass, covering:

- LLM provider configuration
- Advanced mapping inference
- Script generation
- Workflow orchestration
- Excel analysis
- Pattern recognition
- Validation logic

## Performance Tips

- Use local models (Ollama) for development to avoid API costs
- Cache LLM responses to reuse analysis results
- Filter codelists by availability to reduce search space
- Adjust fuzzy matching thresholds based on data quality
- Process multiple files in batch when possible

## Troubleshooting

### Google API Key Issues

The Google API key must be set before importing SDMXerWizard:

```julia
# Correct - set key before import
ENV["GOOGLE_API_KEY"] = "your-key"
using SDMXerWizard

# Wrong - setting key after import is too late
using SDMXerWizard
ENV["GOOGLE_API_KEY"] = "your-key"
```

### Ollama Connection

Ensure Ollama is running:

```bash
ollama serve
ollama list  # Check available models
```

### API Rate Limits

For cloud providers, implement retry logic:

```julia
for attempt in 1:3
    try
        result = generate_transformation_script(...)
        break
    catch e
        if occursin("rate limit", string(e))
            sleep(2^attempt)
        else
            rethrow(e)
        end
    end
end
```

## Contributing

Contributions welcome! Please ensure:

1. All tests pass
2. New features include tests
3. LLM calls are mockable for testing
4. Documentation is updated

## License

MIT License - see [LICENSE](LICENSE) file for details.

## See Also

- [SDMXer.jl](https://github.com/Baffelan/SDMXer.jl) - Core SDMX processing functionality
- [Tidier.jl](https://github.com/TidierOrg/Tidier.jl) - Data transformation framework
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - LLM integration backend
