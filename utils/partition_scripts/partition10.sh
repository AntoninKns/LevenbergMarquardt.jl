#!/bin/bash

julia --project -E 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.resolve()'
julia --project benchmark/distributed_benchmarks.jl 10