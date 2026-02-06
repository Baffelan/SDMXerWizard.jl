#!/usr/bin/env julia
# CI/CD setup script for SDMXerWizard.jl with unregistered SDMXer.jl dependency

using Pkg

# Activate the project
Pkg.activate(".")

# Add SDMXer.jl from GitHub since it's not registered
println("Adding SDMXer.jl from GitHub...")
try
    Pkg.add(url="https://github.com/Baffelan/SDMXer.jl")
catch e
    # If already added, just update it
    println("SDMXer.jl might already be added, updating...")
    Pkg.update("SDMXer")
end

# Instantiate to get all other dependencies
println("Installing dependencies...")
Pkg.instantiate()

# Precompile
println("Precompiling packages...")
Pkg.precompile()

println("Setup complete!")
