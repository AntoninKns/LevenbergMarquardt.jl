export GPUADSolver

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
