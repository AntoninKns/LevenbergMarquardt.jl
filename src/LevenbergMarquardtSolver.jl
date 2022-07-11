export LMSolver

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolver(n, m, S)
may be used in order to create these vectors.
"""
mutable struct LMSolver{T,S}

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

  in_solver :: KrylovSolver

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

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S}(x, Fx, Fxp, xp, Fxm, rows, cols, vals, Jv, Jtv, Ju, Jtu, in_solver, stats)

    return solver
  end
end