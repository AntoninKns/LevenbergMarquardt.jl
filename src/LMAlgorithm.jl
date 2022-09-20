export levenberg_marquardt, levenberg_marquardt!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt(model; AD = false, kwargs...)
  # Adapting the solver depending on automatic differentiation or not
  if AD
    solver = LMSolverAD(model)
  else
    solver = LMSolver(model)
  end
  levenberg_marquardt!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt!(solver    :: AbstractLMSolver{T,S,ST},
                              model     :: AbstractNLSModel;
                              TR        :: Bool = false,
                              λ         :: T = zero(T),
                              Δ         :: T = T(1e4),
                              η₁        :: T = T(eps(T)^(1/4)),
                              η₂        :: T = T(0.99),
                              σ₁        :: T = T(10.0),
                              σ₂        :: T = T(0.1),
                              max_eval  :: Int = 2500,
                              λmin      :: T = T(1e-1),
                              restol    :: T = T(eps(T)^(1/3)),
                              res_rtol  :: T = T(eps(T)^(1/3)),
                              atol      :: T = zero(T), 
                              rtol      :: T = T(eps(T)^(1/3)),
                              in_axtol  :: T = zero(T),
                              in_btol   :: T = zero(T),
                              in_atol   :: T = zero(T),
                              in_rtol   :: T = T(eps(T)^(1/3)),
                              in_etol   :: T = zero(T),
                              in_itmax  :: Int = 0,
                              in_conlim :: T = 1/√eps(T),
                              verbose   :: Bool = true,
                              logging   :: IO = stdout) where {T,S,ST}

  # Set up variables from the solver to avoid allocations
  x, Fx, Fxp, xp, Fxm = solver.x, solver.Fx, solver.Fxp, solver.xp, solver.Fxm
  Jv, Jtv, Ju, Jtu = solver.Jv, solver.Jtv, solver.Ju, solver.Jtu
  in_solver = solver.in_solver
  x .= model.meta.x0

  # Set up the initial value of the residual and Jacobian at the starting point
  residual!(model, x, Fx)

  Jx = set_jac_op_residual!(model, solver, T, S, ST, x, Jv, Jtv)

  # Calculate initiale rNorm and ArNorm values
  rNorm = rNorm0 = norm(Fx)
  mul!(Jtu, Jx', Fx)
  ArNorm = ArNorm0 = norm(Jtu)

  solver.stats.rNorm0 = rNorm0
  solver.stats.ArNorm0 = ArNorm0

  # Set up initial parameters
  iter = 0
  start_time = time()
  optimal_cond = atol + rtol*ArNorm0
  optimal_res = restol + res_rtol*rNorm0
  TR ? param = Δ : param = λ

  optimal = false
  small_residual = false
  tired = false

  # Header log line
  verbose && (levenberg_marquardt_log_header(logging, model, TR, param, η₁, η₂, σ₁, σ₂, max_eval,
                                              λmin, restol, res_rtol, atol, rtol, in_rtol,
                                              in_itmax, in_conlim))
  verbose && (levenberg_marquardt_log_row(logging, iter, rNorm, ArNorm, zero(T), param, zero(T), 
                                          zero(T), zero(T), zero(T), "null", 0, zero(T), 
                                          neval_jprod_residual(model)))

  while !(optimal || small_residual || tired)

    # Time of the step for the log
    start_step_time = time()

    # Solve the subproblem min ‖Jx*d + Fx‖^2
    Fxm .= Fx
    Fxm .*= -1
    in_solver = solve_sub_problem!(in_solver, Jx, Fxm, TR, param,
                                  in_axtol, in_btol, in_atol, in_rtol,
                                  in_etol, in_itmax, in_conlim)

    # Calculate ‖d‖, xk+1, F(xk+1) and ‖F(xk+1)‖
    d = in_solver.x
    dNorm = in_solver.stats.xNorm
    xp .= x .+ d
    Fxp = residual!(model, xp, Fxp)
    rNormp = norm(Fxp)

    println(dNorm)
    println(rNormp)

    # Test the quality of the step 
    # ρ = (‖F(xk)‖² - ‖F(xk+1)‖²) / (‖F(xk)‖² - ‖J(xk)*d + F(xk)‖² - λ‖d‖²)
    mul!(Ju, Jx, d)
    Ju .= Ju .+ Fx
    normJu = norm(Ju)
    rNorm² = rNorm^2
    Pred = (rNorm² - (normJu^2 + param^2*dNorm^2))/2
    Ared = (rNorm² - rNormp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step, update the step and/or the parameters
    if ρ < η₁

      # If the quality of the step is under a certain threshold
      # Adapt λ or Δ to ensure a better next step
      param = bad_step_update!(param, TR, σ₁, λmin)

    else

      # If the step is good enough we accept it and Update
      # x, J(x), F(x), ‖F(x)‖ and ‖J(x)ᵀF(x)‖
      x .= xp
      Jx = update_jac_op_residual!(model, solver, T, S, ST, x, Jv, Jtv)
      Fx .= Fxp
      rNorm = rNormp
      mul!(Jtu, Jx', Fx)
      ArNorm = norm(Jtu)

      if ρ > η₂

        # If the quality of the step is above a certain threshold
        # Loosen λ or Δ to try to find a bigger step
        param = very_good_step_update!(param, σ₂)
      end
      
      # In certains versions of Levenberg Marquardt
      # Some parameters need to be updated in case of a good step
      param = good_step_update!(param, TR, λmin, T)
    end

    # Update logging information
    verbose && (inner_status = change_stats(in_solver.stats.status))
    iter += 1
    solver.stats.inner_iter += in_solver.stats.niter
    step_time = time()-start_step_time
    verbose && (levenberg_marquardt_log_row(logging, iter, rNorm, ArNorm, dNorm, param, Ared, 
                                            Pred, ρ, in_solver.stats.Acond, inner_status, 
                                            in_solver.stats.niter, step_time, 
                                            neval_jprod_residual(model)))
    (logging != stdout) && flush(logging)

    # Update stopping conditions
    optimal = ArNorm < optimal_cond
    tired = neval_residual(model) > max_eval
    small_residual = rNorm < optimal_res

  end

  # Update solver stats
  elapsed_time = time()-start_time
  solver.stats.iter = iter
  solver.stats.elapsed_time = elapsed_time
  solver.stats.rNorm = rNorm
  solver.stats.ArNorm = ArNorm
  solver.stats.solution .= x

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
  solver.stats.status = status

  return solver
end
