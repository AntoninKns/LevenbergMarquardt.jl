export levenberg_marquardt_MINRES, levenberg_marquardt_MINRES!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_MINRES(model; kwargs...)
  # Adapting the solver depending on automatic differentiation or not
  solver = LMSolverMINRES(model)
  solver = levenberg_marquardt_MINRES!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_MINRES!(solver    :: AbstractLMSolver{T,S,ST},
                              model     :: AbstractNLSModel;
                              TR        :: Bool = false,
                              λ         :: T = T(1.),
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
                              in_rtol   :: T = T(eps(T)),
                              in_etol   :: T = zero(T),
                              in_itmax  :: Int = 0,
                              in_conlim :: T = 1/√eps(T),
                              verbose   :: Bool = true,
                              logging   :: IO = stdout) where {T,S,ST}

  # Set up variables from the solver to avoid allocations
  x, Fx, Fxp, xp, Fxm = model.meta.x0, solver.Fx, solver.Fxp, solver.xp, solver.Fxm
  Ju, Jtu = solver.Ju, solver.Jtu
  d = solver.d

  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  # Set up the initial value of the residual and Jacobian at the starting point
  residualMINRES!(model, x, Fx)

  A = augmented_matrix_MINRES(model, solver, x, λ)

  # Calculate initiale rNorm and ArNorm values
  rNorm = rNorm0 = norm(Fx)
  mul!(Jtu, A, Fx)
  @views ArNorm = ArNorm0 = norm(Jtu[m+1:m+n])

  solver.stats.rNorm0 = rNorm0
  solver.stats.ArNorm0 = ArNorm0

  # Set up initial parameters
  iter = 0
  start_time = time()
  optimal_cond = atol + rtol*ArNorm0
  optimal_res = restol + res_rtol*rNorm0

  optimal = false
  small_residual = false
  tired = false

  # Header log line
  verbose && (levenberg_marquardt_log_header(logging, model, TR, λ, η₁, η₂, σ₁, σ₂, max_eval,
                                              λmin, restol, res_rtol, atol, rtol, in_rtol,
                                              in_itmax, in_conlim))
  verbose && (levenberg_marquardt_log_row(logging, iter, (rNorm^2)/2, ArNorm, zero(T), λ, zero(T), 
                                          zero(T), zero(T), zero(T), "null", 0, zero(T), 
                                          neval_jprod_residual(model)))

  while !(optimal || small_residual || tired)

    # Time of the step for the log
    start_step_time = time()

    # Solve the subproblem min ‖Jx*d + Fx‖^2
    Fxm .= Fx
    Fxm .*= -1
    P = ldl(Symmetric(triu(A), :U))
    P.D .= abs.(P.D)
    d, stats = minres(A, Fxm, M=P, rtol=in_rtol, itmax=in_itmax, conlim=in_conlim, ldiv=true)
    @views d2 = d[m+1:m+n]
    
    # Calculate ‖d‖, xk+1, F(xk+1) and ‖F(xk+1)‖
    dNorm = norm(d2)
    xp .= x .+ d2
    residualMINRES!(model, xp, Fxp)
    rNormp = norm(Fxp)

    # Test the quality of the step
    # ρ = (‖F(xk)‖² - ‖F(xk+1)‖²) / (‖F(xk)‖² - ‖J(xk)*d + F(xk)‖² - (λ‖d‖)²)
    @views fill!(d[1:m], zero(T))
    mul!(Ju, A, d)
    @views Ju[1:m] .= Ju[1:m] .+ Fx[1:m]
    normJu = norm(Ju[1:m])
    rNorm² = rNorm^2
    Pred = (rNorm² - (normJu^2 + λ^2*dNorm^2))/2
    Ared = (rNorm² - rNormp^2)/2
    ρ = Ared/Pred

#=     println(rNorm²)
    println(normJu^2)
    println(λ^2*dNorm^2)
    println((normJu^2 + λ^2*dNorm^2))
    println(rNormp^2) =#

    # Depending on the quality of the step, update the step and/or the parameters
    if ρ < η₁

      # If the quality of the step is under a certain threshold
      # Adapt λ or Δ to ensure a better next step
      λ = bad_step_update!(λ, true, σ₁, λmin)
      A = update_jacobian_MINRES(model, solver, λ)

    else

      if ρ > η₂

        # If the quality of the step is above a certain threshold
        # Loosen λ or Δ to try to find a bigger step
        λ = very_good_step_update!(λ, σ₂)
      end
      
      # In certains versions of Levenberg Marquardt
      # Some parameters need to be updated in case of a good step
      λ = good_step_update!(λ, true, λmin, T)

      # If the step is good enough we accept it and Update
      # x, J(x), F(x), ‖F(x)‖ and ‖J(x)ᵀF(x)‖
      x .= xp
      A = augmented_matrix_MINRES(model, solver, x, λ)
      Fx .= Fxp
      rNorm = rNormp
      @views mul!(Jtu, A, Fx)
      ArNorm = norm(Jtu)
    end

    # Update logging information
    verbose && (inner_status = change_stats(stats.status))
    iter += 1
    solver.stats.inner_iter += stats.niter
    step_time = time()-start_step_time
    verbose && (levenberg_marquardt_log_row(logging, iter, (rNorm^2)/2, ArNorm, dNorm, λ, Ared, 
                                            Pred, ρ, 0., inner_status, 
                                            stats.niter, step_time, 
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
