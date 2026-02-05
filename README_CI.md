# CI/CD Setup for SDMXLLM.jl

## Important: Unregistered Dependency

SDMXLLM.jl depends on SDMX.jl, which is not yet registered in the Julia General registry. It's available at https://github.com/Baffelan/SDMX.jl

## Local Development

For local development, the package should already be configured correctly:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## CI/CD Environment Setup

For CI/CD environments (GitHub Actions, GitLab CI, etc.), you need to explicitly add SDMX.jl from GitHub:

```julia
using Pkg
Pkg.activate(".")
# Add unregistered dependency
Pkg.add(url="https://github.com/Baffelan/SDMX.jl")
# Install other dependencies
Pkg.instantiate()
# Run tests
Pkg.test()
```

## GitHub Actions

The `.github/workflows/CI.yml` file has been configured to handle this automatically.

## Manual CI Setup

If you're setting up CI manually, use the provided `ci_setup.jl` script:

```bash
julia ci_setup.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## Docker/Container Setup

For containerized environments:

```dockerfile
FROM julia:1.11

WORKDIR /app
COPY . .

RUN julia -e 'using Pkg; \
    Pkg.activate("."); \
    Pkg.add(url="https://github.com/Baffelan/SDMX.jl"); \
    Pkg.instantiate(); \
    Pkg.precompile()'

CMD ["julia", "--project=.", "-e", "using Pkg; Pkg.test()"]
```

## Troubleshooting

If you encounter "expected package to be registered" errors:
1. Ensure you're adding SDMX.jl via URL, not just by name
2. Clear any cached manifests that might reference a local path
3. Use `Pkg.rm("SDMX")` followed by `Pkg.add(url="https://github.com/Baffelan/SDMX.jl")` to reset