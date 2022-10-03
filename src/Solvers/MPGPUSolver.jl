export MPGPUSolver

mutable struct MPGPUSolver{T,S,ST}

  F32Solver :: AbstractLMSolver
  F64Solver :: AbstractLMSolver

  function MPGPUSolver(model, precisions :: Dict)
  
    if precisions["F32"]
      F32Solver = GPUSolver(model, T = Float32, S = Vector{Float32})
    end

    if precisions["F64"]
      F64Solver = GPUSolver(model, T = Float64, S = Vector{Float64})
    end

    T = eltype(F64Solver.x)
    S = typeof(F64Solver.x)
    ST = typeof(F64Solver.in_solver)
    solver = new{T,S,ST}(F32Solver, F64Solver)

    return solver
  end
end
