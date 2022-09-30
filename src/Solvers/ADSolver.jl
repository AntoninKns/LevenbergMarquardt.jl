export ADSolver

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolverAD(n, m, S)
may be used in order to create these vectors.
"""
mutable struct ADSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S
  d :: S

  Jv :: S
  Jtv :: S

  Ju :: S
  Jtu :: S

  Jx :: LinearOperator{T}

  in_solver :: ST

  TR :: Bool
  λ :: T
  Δ :: T
  λmin :: T

  stats :: LMStats{T,S}

  function ADSolver(model)
  
    x = similar(model.meta.x0)
    m = model.nls_meta.nequ
    n = model.meta.nvar

    T = eltype(x)
    S = typeof(x)

    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    Fxm = similar(x, m)
    d = similar(x, n)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    Ju = similar(x, m)
    Jtu = similar(x, n)

    Jx = opEye(n)

    in_solver = LsmrSolver(m, n, S)

    TR = false
    λ = zero(T)
    Δ = zero(T)
    λmin = zero(T)

    ST = typeof(in_solver)

    stats = LMStats{T,S}(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d, Jv, Jtv, Ju, Jtu, Jx, in_solver, TR, λ, Δ, λmin, stats)

    return solver
  end
end