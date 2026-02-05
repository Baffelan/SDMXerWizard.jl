#!/usr/bin/env julia
# CI/CD setup script for SDMXLLM.jl with unregistered SDMX.jl dependency

using Pkg

# Activate the project
Pkg.activate(".")

# Add SDMX.jl from GitHub since it's not registered
println("Adding SDMX.jl from GitHub...")
try
    Pkg.add(url="https://github.com/Baffelan/SDMX.jl")
catch e
    # If already added, just update it
    println("SDMX.jl might already be added, updating...")
    Pkg.update("SDMX")
end

# Instantiate to get all other dependencies
println("Installing dependencies...")
Pkg.instantiate()

# Precompile
println("Precompiling packages...")
Pkg.precompile()

println("Setup complete!")