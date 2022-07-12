export LMSolverAD

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolverAD(n, m, S)
may be used in order to create these vectors.
"""
mutable struct LMSolverAD{T,S}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S

  Jv :: S
  Jtv :: S

  Ju :: S
  Jtu :: S

  in_solver :: KrylovSolver

  stats :: LMStatsAD{T,S}

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

    stats = LMStatsAD{T,S}(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S}(x, Fx, Fxp, xp, Fxm, Jv, Jtv, Ju, Jtu, in_solver, stats)

    return solver
  end
end