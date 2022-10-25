export GPUSolver

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
  Fxm :: S
  d :: S

  GPUFx :: CuVector{T}
  GPUFxp :: CuVector{T}
  GPUFxm :: CuVector{T}
  GPUd :: CuVector{T}

  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: S

  GPUrows :: CuVector{Int32}
  GPUcols :: CuVector{Int32}
  GPUvals :: CuVector{T}

  Jv :: S
  Jtv :: S

  GPUJv :: CuVector{T}
  GPUJtv :: CuVector{T}

  Ju :: S
  Jtu :: S

  Jx :: LinearOperator{T}
  GPUJx :: CuSparseMatrixCOO{T, Int32}

  in_solver :: ST

  TR :: Bool
  λ :: T
  Δ :: T
  λmin :: T

  stats :: LMStats{T,S}

  function GPUSolver(model;
                    T = eltype(model.meta.x0), 
                    S = typeof(model.meta.x0))
  
    m = model.nls_meta.nequ
    n = model.meta.nvar
    nnzj = model.nls_meta.nnzj

    x = S(undef, n)
    Fx = similar(x, m)
    Fxp = similar(x, m)
    xp = similar(x, n)
    Fxm = similar(x, m)
    d = similar(x, n)
    
    GPUFx = CuVector{T}(undef, m)
    GPUFxp = CuVector{T}(undef, m)
    GPUFxm = CuVector{T}(undef, m)
    GPUd = CuVector{T}(undef, n)

    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    vals = similar(x, nnzj)

    GPUrows = CuVector{Int32}(undef, nnzj)
    GPUcols = CuVector{Int32}(undef, nnzj)
    GPUvals = CuVector{T}(undef, nnzj)

    Jv = similar(x, m)
    Jtv = similar(x, n)

    GPUJv = CuVector{T}(undef, m)
    GPUJtv = CuVector{T}(undef, n)

    Jx = opEye(T, n)
    GPUJx = sparse(GPUrows, GPUcols, GPUvals, m, n, fmt=:coo)

    Ju = similar(x, m)
    Jtu = similar(x, n)

    in_solver = LsmrSolver(m, n, CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer})

    ST = typeof(in_solver)

    TR = false
    λ = zero(T)
    Δ = zero(T)
    λmin = zero(T)

    stats = LMStats(model, :unknown, similar(x), zero(T), zero(T), zero(T), zero(T), 0, 0, 0.)

    solver = new{T,S,ST}(x, Fx, Fxp, xp, Fxm, d,
                         GPUFx, GPUFxp, GPUFxm, GPUd,
                         rows, cols, vals,
                         GPUrows, GPUcols, GPUvals,
                         Jv, Jtv,
                         GPUJv, GPUJtv,
                         Ju, Jtu,
                         Jx, GPUJx,
                         in_solver,
                         TR, λ, Δ, λmin,
                         stats)

    return solver
  end
end
