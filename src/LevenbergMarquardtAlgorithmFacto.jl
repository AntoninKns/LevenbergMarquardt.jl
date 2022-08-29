export levenberg_marquardt_facto, levenberg_marquardt_facto!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_facto(model; kwargs...)
  # Adapting the solver depending on automatic differentiation or not
  solver = LMSolverFacto(model)
  levenberg_marquardt_facto!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt_facto!(solver    :: AbstractLMSolver{T,S,ST},
                              model     :: AbstractNLSModel;
                              TR        :: Bool = false,
                              λ         :: T = T(1.),
                              Δ         :: T = T(1e4),
                              η₁        :: T = eps(T)^(1/4),
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
  x, Fx, Fxp, xp, Fxm = model.meta.x0, solver.Fx, solver.Fxp, solver.xp, solver.Fxm
  Ju, Jtu = solver.Ju, solver.Jtu
  d1 = solver.d

  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  # Set up the initial value of the residual and Jacobian at the starting point
  residual!(model, x, view(Fx, 1:m))
  fill!(view(Fx, m+1:m+n), zero(T))

  @views jac_rows = solver.rows[1:nnzj]
  @views jac_cols = solver.cols[1:nnzj]
  @views jac_vals = solver.vals[1:nnzj]
  jac_structure_residual!(model, jac_rows, jac_cols)
  jac_coord_residual!(model, x, jac_vals)
  @views solver.rows[nnzj+1:nnzj+n] = m+1:m+n
  @views solver.cols[nnzj+1:nnzj+n] = 1:n
  @views fill!(solver.vals[nnzj+1:nnzj+n], √λ)
  A = sparse(solver.rows, solver.cols, solver.vals)

  # Calculate initiale rNorm and ArNorm values
  rNorm = rNorm0 = norm(Fx)
  mul!(Jtu, A', Fx)
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
  verbose && (levenberg_marquardt_log_row(logging, iter, (rNorm^2)/2, ArNorm, zero(T), param, zero(T), 
                                          zero(T), zero(T), zero(T), "null", 0, zero(T), 
                                          neval_jprod_residual(model)))

  while !(optimal || small_residual || tired)

    # Time of the step for the log
    start_step_time = time()

    # Solve the subproblem min ‖Jx*d + Fx‖^2
    Fxm .= Fx
    Fxm .*= -1
    solution = qr(A)
    mul!(d1, solution.Q', Fxm[solution.prow])
    @views d2 = d1[1:n]
    ldiv!(LinearAlgebra.UpperTriangular(solution.R), d2)
    @views d2[solution.pcol] .= d2
    
    # Calculate ‖d‖, xk+1, F(xk+1) and ‖F(xk+1)‖
    d = d2
    dNorm = norm(d)
    xp .= x .+ d
    residual!(model, xp, view(Fxp, 1:m))
    fill!(view(Fxp, m+1:m+n), zero(T))
    rNormp = norm(Fxp)

    # Test the quality of the step 
    # ρ = (‖F(xk)‖² - ‖F(xk+1)‖²) / (‖F(xk)‖² - ‖J(xk)*d + F(xk)‖² - λ‖d‖²)
    @views Jx = sparse(solver.rows[1:nnzj], solver.cols[1:nnzj], solver.vals[1:nnzj])
    @views Ju = Ju[1:m]
    mul!(Ju, Jx, d)
    @views Ju .= Ju .+ Fx[1:m]
    normJu = norm(Ju)
    rNorm² = rNorm^2
    Pred = (rNorm² - (normJu^2 + λ^2*dNorm^2))/2
    Ared = (rNorm² - rNormp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step, update the step and/or the parameters
    if ρ < η₁ || (Ared < 0 && Pred < 0)

      # If the quality of the step is under a certain threshold
      # Adapt λ or Δ to ensure a better next step
      param = bad_step_update!(param, true, σ₁, λmin)
      @views solver.rows[nnzj+1:nnzj+n] = m+1:m+n
      @views solver.cols[nnzj+1:nnzj+n] = 1:n
      @views fill!(solver.vals[nnzj+1:nnzj+n], param)
      A = sparse(solver.rows, solver.cols, solver.vals)
      @views Jx = sparse(solver.rows[1:nnzj], solver.cols[1:nnzj], solver.vals[1:nnzj])

    else

      if ρ > η₂

        # If the quality of the step is above a certain threshold
        # Loosen λ or Δ to try to find a bigger step
        param = very_good_step_update!(param, σ₂)
      end
      
      # In certains versions of Levenberg Marquardt
      # Some parameters need to be updated in case of a good step
      param = good_step_update!(param, true, λmin, T)

      # If the step is good enough we accept it and Update
      # x, J(x), F(x), ‖F(x)‖ and ‖J(x)ᵀF(x)‖
      x .= xp
      @views jac_rows = solver.rows[1:nnzj]
      @views jac_cols = solver.cols[1:nnzj]
      @views jac_vals = solver.vals[1:nnzj]
      jac_structure_residual!(model, jac_rows, jac_cols)
      jac_coord_residual!(model, x, jac_vals)
      @views solver.rows[nnzj+1:nnzj+n] = m+1:m+n
      @views solver.cols[nnzj+1:nnzj+n] = 1:n
      @views fill!(solver.vals[nnzj+1:nnzj+n], param)
      A = sparse(solver.rows, solver.cols, solver.vals)
      @views Jx = sparse(solver.rows[1:nnzj], solver.cols[1:nnzj], solver.vals[1:nnzj])
      Fx .= Fxp
      rNorm = rNormp
      @views mul!(Jtu, Jx', Fx[1:m])
      ArNorm = norm(Jtu)
    end

    # Update logging information
    verbose && (inner_status = "QR")
    iter += 1
    solver.stats.inner_iter += 1
    step_time = time()-start_step_time
    verbose && (levenberg_marquardt_log_row(logging, iter, (rNorm^2)/2, ArNorm, dNorm, param, Ared, 
                                            Pred, ρ, 0., inner_status, 
                                            1, step_time, 
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