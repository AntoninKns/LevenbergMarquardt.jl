export levenberg_marquardt_AD, levenberg_marquardt_AD!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_AD(model; kwargs...)
  solver = LMSolverAD(model)
  levenberg_marquardt_AD!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_AD!(solver    :: LMSolverAD{T,S}, 
                              model     :: AbstractNLSModel;
                              λ         :: T = zero(T),
                              η₁        :: T = eps(T)^(1/4), 
                              η₂        :: T = T(0.99),
                              σ₁        :: T = T(10.0), 
                              σ₂        :: T = T(0.1),
                              max_eval  :: Int = 10_000,
                              λmin      :: T = T(1e-1),
                              restol    :: T = T(eps(T)^(1/3)),
                              atol      :: T = zero(T), 
                              rtol      :: T = T(eps(T)^(1/3)),
                              in_axtol  :: T = zero(T),
                              in_btol   :: T = zero(T),
                              in_atol   :: T = zero(T),
                              in_rtol   :: T = zero(T),
                              in_etol   :: T = zero(T),
                              in_itmax  :: Int = 0,
                              in_conlim :: T = 1/√eps(T),
                              verbose   :: Bool = true) where {T, S}

  # Set up the initial value of the residual and Jacobin at the starting point
  x, Fx, Fxp, xp, Fxm = model.meta.x0, solver.Fx, solver.Fxp, solver.xp, solver.Fxm
  Jv, Jtv, Ju, Jtu = solver.Jv, solver.Jtv, solver.Ju, solver.Jtu
  in_solver = solver.in_solver

  residual!(model, x, Fx)

  Jx = jac_op_residual!(model, x, Jv, Jtv)

  rNorm = rNorm0 = norm(Fx)
  mul!(Jtu, Jx', Fx)
  ArNorm = ArNorm0 = norm(Jtu)

  solver.stats.rNorm0 = rNorm0
  solver.stats.ArNorm0 = ArNorm0

  # Set up initial parameters
  iter = 0
  start_time = time()
  optimal_cond = atol + rtol*ArNorm0

  optimal = false
  small_residual = false
  tired = false
  
  verbose && (levenberg_marquardt_log_header(model))

  while !(optimal || small_residual || tired )

    # Solve the subproblem
    Fxm .= Fx
    Fxm .*= -1
    in_solver = lsmr!(in_solver, Jx, Fxm, 
                      λ = λ, 
                      axtol = in_axtol,
                      btol = in_btol,
                      atol = in_atol,
                      rtol = in_rtol,
                      etol = in_etol,
                      itmax = in_itmax,
                      conlim = in_conlim)

    d = in_solver.x
    dNorm = norm(d)
    xp .= x .+ d
    Fxp = residual!(model, xp, Fxp)
    rNormp = norm(Fxp)

    # Test the quality of the step
    mul!(Ju, Jx, d)
    Ju .= Ju .+ Fx
    normJu = norm(Ju)
    rNorm² = rNorm^2
    Pred = (rNorm² - (normJu^2 + λ^2*dNorm^2))/2
    Ared = (rNorm² - rNormp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step we update the step and/or the parameters
    if ρ < η₁
      if λ < λmin
        λ = λmin
      else
        λ = σ₁ * λ
      end
    else
      x .= xp
      Jx = jac_op_residual!(model, x, Jv, Jtv)
      Fx .= Fxp
      rNorm = rNormp
      mul!(Jtu, Jx', Fx)
      ArNorm = norm(Jtu)
      if ρ > η₂
        λ = σ₂ * λ
      end
      if λ < λmin
        λ = zero(T)
      end
    end

    # Update logging information
    verbose && (inner_status = change_stats(in_solver.stats.status))
    iter += 1
    solver.stats.inner_iter += in_solver.stats.niter
    verbose && (levenberg_marquardt_log_row(iter, (rNorm^2)/2, ArNorm, dNorm, λ, Ared, Pred, ρ, inner_status, in_solver.stats.niter, neval_jprod_residual(model)))

    # Update stopping conditions
    optimal = ArNorm < optimal_cond
    tired = neval_residual(model) > max_eval
    small_residual = rNorm < restol

  end

  # Update solver stats
  elapsed_time = time()-start_time
  solver.stats.iter = iter
  solver.stats.elapsed_time = elapsed_time
  
  # Update solver status
  if optimal 
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif tired
    status = :max_iter
  else
    status = :unknown
  end

  return solver
end