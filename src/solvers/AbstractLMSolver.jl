export AbstractLMSolver

"Abstract type for using Levenberg Marquardt solvers in-place"
abstract type AbstractLMSolver{T,S,ST} end

include("ADSolver.jl")
include("GPUSolver.jl")
include("LDLSolver.jl")
include("LMSolver.jl")
include("MINRESSolver.jl")
include("MPGPUSolver.jl")
include("MPSolver.jl")
include("CGSolver.jl")
include("SCHURSolver.jl")
