export MPSolver

mutable struct MPSolver{T,S,ST}

  F32Solver :: AbstractLMSolver
  F64Solver :: AbstractLMSolver

  function LMMPSolver(model; F32 = false, F64 = true)
  
    if F32
      F32Solver = LMSolver(model, T = Float32, S = Vector{Float32})
    end

    if F64
      F64Solver = LMSolver(model, T = Float64, S = Vector{Float64})
    end

    T = eltype(F64Solver.x)
    S = typeof(F64Solver.x)
    ST = typeof(F64Solver.in_solver)
    solver = new{T,S,ST}(F32Solver, F64Solver)

    return solver
  end
end

mutable struct MINRESSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S
  d :: S

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  Ju :: S
  Jtu :: S

  stats :: LMStats{T,S}

  function LMSolverMINRES(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj
    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m+n)
    Fxp = similar(x, m+n)
    xp = similar(x, n)
    Fxm = similar(x, m+n)
    d = similar(x, m+n)

    rows = Vector{Int}(undef, m+2*nnzj+n)
    cols = Vector{Int}(undef, m+2*nnzj+n)
    vals = similar(x, m+2*nnzj+n)

    Ju = similar(x, m+n)
    Jtu = similar(x, m+n)

    ST = Float64

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d, rows, cols, vals, Ju, Jtu, stats)

    return solver
  end
end