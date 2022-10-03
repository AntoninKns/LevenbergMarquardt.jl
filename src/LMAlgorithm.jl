export levenberg_marquardt, levenberg_marquardt!

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt(model :: AbstractNLSModel; version :: Symbol = :DEFAULT, precisions :: Dict = Dict("F32" => true, "F64" => true), kwargs...)
  # Adapting the solver depending on the wanted version
  if version == :DEFAULT
    generic_solver = LMSolver(model)
  elseif version == :GPU
    CUDA.allowscalar(false)
    generic_solver = GPUSolver(model)
  elseif version == :LDL
    generic_solver = LDLSolver(model)
  elseif version == :MP
    generic_solver = MPSolver(model, precisions)
  elseif version == :MPGPU
    CUDA.allowscalar(false)
    generic_solver = MPGPUSolver(model, precisions)
  elseif version == :MINRES
    generic_solver = MINRESSolver(model)
  else
    error("Could not recognize Levenberg-Marquardt version. Available versions are given in the docstring of the function.")
  end
  levenberg_marquardt!(generic_solver, model; kwargs...)
  if version == :MP || version == :MPGPU
    return generic_solver.F64Solver.stats
  else
    return generic_solver.stats
  end
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt(model :: ReverseADNLSModel; version :: Symbol = :DEFAULT, kwargs...)
  # Adapting the solver depending on the wanted version
  if version == :DEFAULT
    solver = ADSolver(model)
  else
    error("Could not recognize Levenberg-Marquardt version. Available versions are given in the docstring of the function.")
  end
  levenberg_marquardt!(solver, model; kwargs...)
  return solver.stats
end

"""
Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt
"""
function levenberg_marquardt!(generic_solver :: Union{AbstractLMSolver{T,S,ST}, MPSolver{T,S,ST}, MPGPUSolver{T,S,ST}},
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
  
  # Set up initial variables contained in the solver
  (x, d, xp, solver) = set_variables!(model, generic_solver, TR, λ, Δ, λmin)

  # Set up the initial value of the residual and Jacobian at the starting point
  residualLM!(model, x, solver)
  set_jac_residual!(model, x, solver)
  
  # Calculate initiale rNorm and ArNorm values
  rNorm = rNorm0 = rNorm!(solver)
  ArNorm = ArNorm0 = ArNorm!(model, solver)

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
  verbose && (levenberg_marquardt_log_header(logging, model, solver, η₁, η₂, σ₁, σ₂, max_eval,
                                              restol, res_rtol, atol, rtol, in_rtol,
                                              in_itmax, in_conlim, Val(solver.TR)))
  verbose && (levenberg_marquardt_log_row(logging, model, solver, iter, rNorm, ArNorm, zero(T), 
                                          zero(T), zero(T), zero(T), "null", 
                                          zero(T), Val(solver.TR)))

  while !(optimal || small_residual || tired)

    # Time of the step for the log
    start_step_time = time()

    # Solve the subproblem min ‖Jx*d + Fx‖^2
    solve_sub_problem!(model, generic_solver, in_axtol, in_btol, in_atol, 
                       in_rtol, in_etol, in_itmax, in_conlim, Val(solver.TR))

    # Calculate ‖d‖, xk+1, F(xk+1) and ‖F(xk+1)‖
    d = step!(model, solver)
    dNorm = norm(d)
    xp .= x .+ d
    residualLMp!(model, xp, solver)
    rNormp = rNormp!(solver)

    # Test the quality of the step 
    # Ared = ‖F(xk)‖² - ‖F(xk+1)‖²
    # Pred = ‖F(xk)‖² - (‖J(xk)*d + F(xk)‖² + λ²‖d‖²)
    # ρ = Ared / Pred
    Ared = ared(solver, rNorm, rNormp)
    Pred = pred(model, solver, rNorm, dNorm, Val(solver.TR))
    ρ = Ared/Pred

    # Depending on the quality of the step, update the step and/or the parameters
    if ρ < η₁

      # If the quality of the step is under a certain threshold
      # Adapt λ or Δ to ensure a better next step
      bad_step_update!(solver, σ₁, Val(solver.TR))
      update_lambda!(model, solver)

    else

      # If the step is good enough we accept it and Update
      # x, J(x), F(x), ‖F(x)‖ and ‖J(x)ᵀF(x)‖
      x .= xp
      update_jac_residual!(model, x, solver)
      residualLM!(model, x, solver)
      rNorm = rNormp
      ArNorm = ArNorm!(model, solver)

      if ρ > η₂

        # If the quality of the step is above a certain threshold
        # Loosen λ or Δ to try to find a bigger step
        very_good_step_update!(solver, σ₂, Val(solver.TR))
      end
      
      # In certains versions of Levenberg Marquardt
      # Some parameters need to be updated in case of a good step
      good_step_update!(solver, T, Val(solver.TR))
    end

    # Update logging information
    verbose && (inner_status = change_stats(solver))
    iter += 1
    step_time = time()-start_step_time
    verbose && (levenberg_marquardt_log_row(logging, model, solver, iter, rNorm, ArNorm, 
                                            dNorm, Ared, Pred, ρ, inner_status, 
                                            step_time, Val(solver.TR)))
    (logging != stdout) && flush(logging)

    # Update stopping conditions
    optimal = ArNorm < optimal_cond
    tired = neval_residual(model) > max_eval
    small_residual = rNorm < optimal_res

  end

  # Update solver stats
  elapsed_time = time() - start_time
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
