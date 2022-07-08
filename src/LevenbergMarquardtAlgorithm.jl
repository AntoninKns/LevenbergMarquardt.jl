export levenberg_marquardt, levenberg_marquardt!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt(model; kwargs...)
  solver = LMSolver(model)
  levenberg_marquardt!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt!(solver :: LMSolver{T}, 
                              model :: AbstractNLSModel;
                              x :: AbstractVector = copy(model.meta.x0),
                              λ :: AbstractFloat = zero(T),
                              η₁ :: AbstractFloat = T(1e-4), 
                              η₂ :: AbstractFloat = T(0.99),
                              σ₁ :: AbstractFloat = T(10.0), 
                              σ₂ :: AbstractFloat = T(0.1),
                              max_eval :: Int = 10_000,
                              λmin :: AbstractFloat = T(1e-1),
                              restol = T(eps(T)^(1/3)),
                              atol = zero(T), 
                              rtol = T(eps(T)^(1/3)),
                              in_axtol :: AbstractFloat = √eps(T),
                              in_btol :: AbstractFloat = √eps(T),
                              in_atol :: AbstractFloat = zero(T),
                              in_rtol :: AbstractFloat = zero(T),
                              in_etol :: AbstractFloat = √eps(T),
                              in_itmax :: Int = 0,
                              in_conlim :: AbstractFloat = 1/√eps(T),
                              verbose :: Bool = true) where T <: AbstractFloat

  # We set up the initial value of the residual and Jacobien based on the starting point
  Fx, Fxp, xp, Fxm = solver.Fx, solver.Fxp, solver.xp, solver.Fxm
  rows, cols, vals = solver.rows, solver.cols, solver.vals
  Jv, Jtv, Ju, Jtu = solver.Jv, solver.Jtv, solver.Ju, solver.Jtu
  m, n, nnzj = model.nls_meta.nequ, model.meta.nvar, model.nls_meta.nnzj
  in_solver = solver.in_solver

  Fx = residual!(model, x, Fx)

  jac_structure_residual!(model, rows, cols)
  jac_coord_residual!(model, x, vals)
  Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)

  rNorm = rNorm0 = BLAS.nrm2(m, Fx, 1)
  Jtu = mul!(Jtu, transpose(Jx), Fx)
  ArNorm = ArNorm0 = BLAS.nrm2(n, Jtu, 1)

  solver.stats.rNorm0 = rNorm0
  solver.stats.ArNorm0 = ArNorm0

  # We set up the initial parameters 
  iter = 0
  start_time = time()

  optimal = false
  small_residual = false
  tired = false
  
  verbose && (levenberg_marquardt_log_header(model))

  while !(optimal || small_residual || tired )

    # We solve the subproblem
    Fxm .= .-Fx
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
    dNorm = BLAS.nrm2(d)
    xp .= x .+ d
    Fxp = residual!(model, xp, Fxp)
    rNormp = BLAS.nrm2(m, Fxp, 1)

    # We test the quality of the state
    Ju = mul!(Ju, Jx, d)
    Ju .= Ju .+ Fx
    normJu = BLAS.nrm2(m, Ju, 1)
    Pred = (rNorm^2 - normJu^2 - λ^2*dNorm^2)/2
    Ared = (rNorm^2 - rNormp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step we update the step and/or the parameters
    if ρ < η₁
      if λ == zero(T)
        λ = λmin
      else
        λ = σ₁ * λ
      end
    else
      x .= xp
      jac_coord_residual!(model, x, vals)
      Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)
      Fx .= Fxp
      rNorm = rNormp
      Jtu = mul!(Jtu, transpose(Jx), Fx)
      ArNorm = BLAS.nrm2(n, Jtu, 1)
      if ρ > η₂
        λ = σ₂ * λ
      end
      if λ < λmin
        λ = zero(T)
      end
    end

    # We update the logging informations
    verbose && (inner_status = change_stats(in_solver.stats.status))
    iter += 1
    solver.stats.inner_iter += in_solver.stats.niter
    verbose && (levenberg_marquardt_log_row(iter, (rNorm^2)/2, ArNorm, dNorm, λ, Ared, Pred, ρ, inner_status, in_solver.stats.niter, neval_jprod_residual(model)))

    # We update the stopping conditions
    optimal = ArNorm < atol + rtol*ArNorm0
    tired = neval_residual(model) > max_eval
    small_residual = rNorm < restol

  end

  # We update the last parameters
  elapsed_time = time()-start_time
  solver.stats.iter = iter
  solver.stats.elapsed_time = elapsed_time
  
  # we update the status of the solver
  if optimal 
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif tired
    status = :max_iter
  else
    status = :unknown
  end

  # We return all the logs in a dedicated structure
  return solver
end
