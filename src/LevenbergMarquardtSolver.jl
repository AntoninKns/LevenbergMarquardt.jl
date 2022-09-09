export AbstractLMSolver, LMSolver, LMSolverAD, LMSolverFacto

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

  function LMSolver(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj
    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    Fxm = similar(x, m)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = similar(x, nnzj)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    Ju = similar(x, m)
    Jtu = similar(x, n)

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
