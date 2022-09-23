export AbstractLMSolver, LMSolver, LMSolverAD, LMSolverFacto, GPUSolver, LMMPSolver, LMSolverGPUAD

"Abstract type for using Levenberg Marquardt solvers in-place"
abstract type AbstractLMSolver{T,S,ST} end

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolver(n, m, S)
may be used in order to create these vectors.
"""
mutable struct LMSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  Jv :: S
  Jtv :: S

  Ju :: S
  Jtu :: S

  in_solver :: ST

  stats :: LMStats{T,S}

  function LMSolver(model; 
                    T = eltype(model.meta.x0), 
                    S = typeof(model.meta.x0))
  
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj

    x = S(undef, n)
    Fx = S(undef, m)
    Fxp = S(undef, m)
    xp = S(undef, n)
    Fxm = S(undef, m)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = S(undef, nnzj)

    Jv = S(undef, m)
    Jtv = S(undef, n)

    Ju = S(undef, m)
    Jtu = S(undef, n)

    in_solver = LsmrSolver(m, n, S)

    ST = typeof(in_solver)

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, rows, cols, vals, Jv, Jtv, Ju, Jtu, in_solver, stats)

    return solver
  end
end

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolverAD(n, m, S)
may be used in order to create these vectors.
"""
mutable struct LMSolverAD{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S

  Jv :: S
  Jtv :: S

  Ju :: S
  Jtu :: S

  in_solver :: ST

  stats :: LMStats{T,S}

  function LMSolverAD(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar

    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    Fxm = similar(x, m)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    Ju = similar(x, m)
    Jtu = similar(x, n)

    in_solver = LsmrSolver(m, n, S)

    ST = typeof(in_solver)

    stats = LMStats{T,S}(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, Jv, Jtv, Ju, Jtu, in_solver, stats)

    return solver
  end
end

mutable struct LMSolverFacto{T,S,ST} <: AbstractLMSolver{T,S,ST}

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

  function LMSolverFacto(model)
  
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

    rows = Vector{Int}(undef, nnzj+n)
    cols = Vector{Int}(undef, nnzj+n)
    vals = similar(x, nnzj+n)

    Ju = similar(x, m+n)
    Jtu = similar(x, n)

    ST = Float64

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d, rows, cols, vals, Ju, Jtu, stats)

    return solver
  end
end

mutable struct LMSolverLDL{T,S,ST} <: AbstractLMSolver{T,S,ST}

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

  function LMSolverLDL(model)
  
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

    rows = Vector{Int}(undef, m+nnzj+n)
    cols = Vector{Int}(undef, m+nnzj+n)
    vals = similar(x, m+nnzj+n)

    Ju = similar(x, m+n)
    Jtu = similar(x, m+n)

    ST = Float64

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d, rows, cols, vals, Ju, Jtu, stats)

    return solver
  end
end

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = GPUSolver(n, m, S)
may be used in order to create these vectors.
"""
mutable struct GPUSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  d :: S

  GPUFx :: CuVector{T}
  GPUFxp :: CuVector{T}
  GPUFxm :: CuVector{T}
  GPUd :: CuVector{T}

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  GPUrows :: CuVector{Int}
  GPUcols :: CuVector{Int}
  GPUvals :: CuVector{T}

  Jv :: S
  Jtv :: S

  GPUJv :: CuVector{T}
  GPUJtv :: CuVector{T}

  Ju :: S
  Jtu :: S

  in_solver :: ST

  stats :: LMStats{T,S}

  function GPUSolver(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj
    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    d = similar(x, n)
    
    GPUFx = CuVector{T}(undef, m)
    GPUFxp = CuVector{T}(undef, m)
    GPUFxm = CuVector{T}(undef, m)
    GPUd = CuVector{T}(undef, n)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = similar(x, nnzj)

    GPUrows = CuVector{Int}(undef, nnzj)
    GPUcols = CuVector{Int}(undef, nnzj)
    GPUvals = CuVector{T}(undef, nnzj)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    GPUJv = CuVector{T}(undef, m)
    GPUJtv = CuVector{T}(undef, n)

    Ju = similar(x, m)
    Jtu = similar(x, n)

    in_solver = LsmrSolver(m, n, CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer})

    ST = typeof(in_solver)

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, d,
                         GPUFx, GPUFxp, GPUFxm, GPUd,
                         rows, cols, vals, 
                         GPUrows, GPUcols, GPUvals,
                         Jv, Jtv, 
                         GPUJv, GPUJtv,
                         Ju, Jtu, 
                         in_solver, 
                         stats)

    return solver
  end
end


mutable struct LMMPSolver{T,S,ST}

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

mutable struct LMSolverMINRES{T,S,ST} <: AbstractLMSolver{T,S,ST}

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

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolverAD(n, m, S)
may be used in order to create these vectors.
"""
mutable struct LMSolverGPUAD{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  d :: S

  GPUx :: CuVector{T}
  GPUFx :: CuVector{T}
  GPUFxp :: CuVector{T}
  GPUFxm :: CuVector{T}
  GPUd :: CuVector{T}

  Jv :: S
  Jtv :: S

  GPUJv :: CuVector{T}
  GPUJtv :: CuVector{T}

  Ju :: S
  Jtu :: S

  GPUJu :: CuVector{T}
  GPUJtu :: CuVector{T}

  in_solver :: ST

  stats :: LMStats{T,S}

  function LMSolverGPUAD(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar

    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    d = similar(x, n)

    GPUx = CuVector{T}(undef, n)
    GPUFx = CuVector{T}(undef, m)
    GPUFxp = CuVector{T}(undef, m)
    GPUFxm = CuVector{T}(undef, m)
    GPUd = CuVector{T}(undef, n)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    GPUJv = CuVector{T}(undef, m)
    GPUJtv = CuVector{T}(undef, n)
  
    Ju = similar(x, m)
    Jtu = similar(x, n)

    GPUJu = CuVector{T}(undef, m)
    GPUJtu = CuVector{T}(undef, n)
  
    in_solver = LsmrSolver(m, n, CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer})
  
    ST = typeof(in_solver)

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, d,
                         GPUx, GPUFx, GPUFxp, GPUFxm, GPUd,
                         Jv, Jtv, 
                         GPUJv, GPUJtv,
                         Ju, Jtu,
                         GPUJu, GPUJtu,
                         in_solver, 
                         stats)

    return solver
  end
end
