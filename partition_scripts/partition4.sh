#!/bin/bash

julia --project -E 'using Pkg; Pkg.update(); Pkg.resolve()'
julia --project benchmark/distributed_benchmarks.jl 4