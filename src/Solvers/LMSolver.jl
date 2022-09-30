export LMSolver

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
  d :: S

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

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
    d = S(undef, n)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = S(undef, nnzj)

    Jv = S(undef, m)
    Jtv = S(undef, n)

    Ju = S(undef, m)
    Jtu = S(undef, n)

    Jx = opEye(n)

    in_solver = LsmrSolver(m, n, S)

    TR = false
    λ = zero(T)
    Δ = zero(T)
    λmin = zero(T)

    ST = typeof(in_solver)

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d, rows, cols, vals, Jv, Jtv, Ju, Jtu, Jx,
                          in_solver, TR, λ, Δ, λmin, stats)

    return solver
  end
end
