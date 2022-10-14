# Algorithm of Levenberg Marquardt based on "AN INEXACT LEVENBERG-MARQUARDT METHOD FOR
# LARGE SPARSE NONLINEAR LEAST SQUARES" from Wright and Holt

export levenberg_marquardt, levenberg_marquardt!

"""
    stats = levenberg_marquardt(model      :: AbstractNLSModel; 
                                which      :: Symbol=:DEFAULT,
                                precisions :: Dict = Dict("F32" => true, "F64" => true),
                                TR         :: Bool = false,
                                λ          :: T = zero(T),
                                Δ          :: T = T(1e4),
                                λmin       :: T = T(1e-1),
                                η₁         :: T = T(eps(T)^(1/4)),
                                η₂         :: T = T(0.99),
                                σ₁         :: T = T(10.0),
                                σ₂         :: T = T(0.1),
                                max_eval   :: Int = 2500,
                                restol     :: T = T(eps(T)^(1/3)),
                                res_rtol   :: T = T(eps(T)^(1/3)),
                                atol       :: T = zero(T), 
                                rtol       :: T = T(eps(T)^(1/3)),
                                in_axtol   :: T = zero(T),
                                in_btol    :: T = zero(T),
                                in_atol    :: T = zero(T),
                                in_rtol    :: T = T(eps(T)^(1/3)),
                                in_etol    :: T = zero(T),
                                in_itmax   :: Int = 0,
                                in_conlim  :: T = 1/√eps(T),
                                verbose    :: Bool = true,
                                logging    :: IO = stdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`S` is `Vector{T}`.
`ST is the type of the internal solver. For which=:DEFAULT, `ST` is `LsmrSolver{T, T, S}`.

`which` determines the version of the LevenbergMarquardt algorithm.

`which = :DEFAULT` solves the subproblem with the LSMR iterative method on CPU in the precision given in the model.

`which = :GPU` solves the subproblem with LSMR iterative method on GPU in the precision given in the model.
The rest of the algorithm runs on CPU. Currently, only Nvidia GPUs running with CUDA are supported.

`which = :LDL` solves the augmented subproblem with LDL factorization in the precision given in the model.

`which = :MP` solves the subproblem with LSMR iterative method following iterative refinement strategy with 
precisions given in `precisions` dictionary. Currently, only 2 precisions are supported.

`which = :MPGPU` is a combination of `:MP` and `:GPU`, solving the subproblem on GPU with iterative refinement strategy.

`which = :MINRES` solves the augmented subproblem with MINRES iterative method using a preconditioner formed
either through LDL factorization or limited LDL factorization. The code runs on CPU in the precision givne in the model.


`precisions` determines the precisions used for the `:MP` and `:MPGPU` versions.

`TR` determines if a regularized or trust region version of the algorithm is used. 
`λ` and `Δ` then determine the initial value of the chosne version.
`λmin` determines the minimum value tha can be attained by `λ` before being set to 0.

`η₁` and `η₂` determine the value for which we consider a step to be either unsuccessful, successsful or very successsful.
`σ₁` and `σ₂` then determine the value by which the parameter (`λ` or `Δ`) is multiplied.

`max_eval`, `restol`, `res_rtol`, `atol` and `rtol` are the stopping criterias of the Levenberg-Marquardt algorithm.

`in_axtol`, `in_btol`, `in_atol`, `in_rtol`, `in_etol`, `in_itmax` and `in_conlim` are the stopping criterias of the subproblem
in the Levenberg-Marquardt algorithm. If a direct method such as `:LDL` is used, the stopping criterias are ignored.
"""
function levenberg_marquardt(model :: AbstractNLSModel; which :: Symbol = :DEFAULT, precisions :: Dict = Dict("F32" => true, "F64" => true), kwargs...)
  # Adapting the solver depending on the wanted version
  if which == :DEFAULT
    generic_solver = LMSolver(model)
  elseif which == :GPU
    # If the code is run on GPU, indexing elemnts are forbidden.
    CUDA.allowscalar(false)
    generic_solver = GPUSolver(model)
  elseif which == :LDL
    generic_solver = LDLSolver(model)
  elseif which == :MP
    generic_solver = MPSolver(model, precisions)
  elseif which == :MPGPU
    # If the code is run on GPU, indexing elemnts are forbidden.
    CUDA.allowscalar(false)
    generic_solver = MPGPUSolver(model, precisions)
  elseif which == :MINRES
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

function levenberg_marquardt(model :: ReverseADNLSModel; which :: Symbol = :DEFAULT, kwargs...)
  # Adapting the solver depending on the wanted version
  if which == :DEFAULT
    solver = ADSolver(model)
  else
    error("Could not recognize Levenberg-Marquardt version. Available versions are given in the docstring of the function.")
  end
  levenberg_marquardt!(solver, model; kwargs...)
  return solver.stats
end

"""
    generic_solver = levenberg_marquardt!(generic_solver::Union{AbstractLMSolver, MPSolver, MPGPUSolver}, model::AbstractNLSModel; kwargs...)

where `kwargs` are keyword arguments of [`levenberg_marquardt`](@ref).

`generic_solver` is either an `AbstractLMSolver` or a wrapper like `MPSolver` or `MPGPUSolver` that contains multiple `AbstractLMSolver`.
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

    # Solve the subproblem min 1/2 ‖Jx*d + Fx‖^2 #a corriger
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
        update_lambda!(model, solver)
      end
      
      # In certains versions of Levenberg Marquardt
      # Some parameters need to be updated in case of a good step
      good_step_update!(solver, T, Val(solver.TR))
      update_lambda!(model, solver)
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
