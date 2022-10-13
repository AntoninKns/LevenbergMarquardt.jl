export LDLSolver

mutable struct LDLSolver{T,S,ST} <: AbstractLMSolver{T,S,ST}

  x :: S
  Fx :: S
  Fxp :: S
  xp :: S
  Fxm :: S
  fulld :: S
  d :: S

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  Ju :: S
  Jtu :: S

  A :: SparseMatrixCSC{T, Int64}

  TR :: Bool
  λ :: T
  Δ :: T
  λmin :: T

  stats :: LMStats{T,S}

  function LDLSolver(model)
  
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
    fulld = similar(x, m+n)
    d = similar(x, n)

    rows = Vector{Int}(undef, m+nnzj+n)
    cols = Vector{Int}(undef, m+nnzj+n)
    vals = similar(x, m+nnzj+n)

    Ju = similar(x, m+n)
    Jtu = similar(x, m+n)

    A = sparse([one(T)], [one(T)], [one(T)])

    TR = false
    λ = one(T)
    Δ = zero(T)
    λmin = zero(T)

    ST = Float64

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, fulld, d, rows, cols, vals, Ju, Jtu, A, TR, λ, Δ, λmin, stats)

    return solver
  end
end