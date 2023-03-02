export SCHURSolver

"""
Type for storing the vectors required by the in-place version of LevenbergMarquardt.
The outer constructor
    solver = LMSolver(n, m, S)
may be used in order to create these vectors.
"""
mutable struct SCHURSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  JtFxm :: S
  d :: S

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  Jx :: SparseMatrixCSC{T, Int64}

  JtJ :: SparseMatrixCSC{T, Int64}
  B :: SparseMatrixCSC{T, Int64}
  C :: SparseMatrixCSC{T, Int64}
  E :: SparseMatrixCSC{T, Int64}
  Schur :: SparseMatrixCSC{T, Int64}

  Jv :: S
  Jtv :: S

  Ju :: S
  Jtu :: S

  TR :: Bool
  λ :: T
  Δ :: T
  λmin :: T

  stats :: LMStats{T,S}

  function SCHURSolver(model; 
                    T = eltype(model.meta.x0), 
                    S = typeof(model.meta.x0))
  
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj
    npnts = model.npnts
    ncams = model.ncams

    x = S(undef, n)
    Fx = S(undef, m)
    Fxp = S(undef, m)
    xp = S(undef, n)
    JtFxm = S(undef, n)
    d = S(undef, n)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = S(undef, nnzj)

    Jx = SparseMatrixCSC{T, Int64}(I, m, n)

    JtJ = SparseMatrixCSC{T, Int64}(I, n, n)
    B = SparseMatrixCSC{T, Int64}(I, 3*npnts, 3*npnts)
    C = SparseMatrixCSC{T, Int64}(I, 9*ncams, 9*ncams)
    E = SparseMatrixCSC{T, Int64}(I, 3*npnts, 9*ncams)
    Schur = SparseMatrixCSC{T, Int64}(I, 9*ncams, 9*ncams)

    Jv = S(undef, m)
    Jtv = S(undef, n)

    Ju = S(undef, m)
    Jtu = S(undef, n)

    TR = false
    λ = zero(T)
    Δ = zero(T)
    λmin = zero(T)

    ST = Float64

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, JtFxm, d, rows, cols, vals, Jx, JtJ, B, C, E, Schur, Jv, Jtv, Ju, Jtu,
                         TR, λ, Δ, λmin, stats)

    return solver
  end
end
