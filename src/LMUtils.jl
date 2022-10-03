"""
Shortens status log of Levenberg Marquardt subproblem
"""
function change_stats(solver :: Union{LMSolver, ADSolver, GPUSolver})
  status = solver.in_solver.stats.status
  if status == "maximum number of iterations exceeded"
    status = "iter"
  elseif status == "condition number seems too large for this machine"
    status = "cond"
  elseif status == "condition number exceeds tolerance"
    status = "cond"
  elseif status == "found approximate minimum least-squares solution"
    status = "ok"
  elseif status == "found approximate zero-residual solution"
    status = "ok"
  elseif status == "truncated forward error small enough"
    status = "err"
  elseif status == "on trust-region boundary"
    status = "TR"
  elseif status == "user-requested exit"
    status = "user"
  else
    status = status
  end
  return status
end

"""
Header of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_header(logging :: IO, model :: AbstractNLSModel, solver :: AbstractLMSolver, η₁ :: AbstractFloat, η₂ :: AbstractFloat, 
                                              σ₁ :: AbstractFloat, σ₂ :: AbstractFloat, max_eval :: Integer, restol :: AbstractFloat, res_rtol :: AbstractFloat, 
                                              atol :: AbstractFloat, rtol :: AbstractFloat, in_rtol :: AbstractFloat, in_itmax :: Integer, in_conlim, :: Val{true})
  @printf(logging, "Solving %s with %d equations and %d variables\n\n", model.meta.name, model.nls_meta.nequ, model.meta.nvar)
  @printf(logging, "Parameters of the solver :\n")
  @printf(logging, "| %1s        : %1.2e | η₁       : %1.2e | η₂        : %1.2e | σ₁   : %1.2e | σ₂   : %1.2e |", "Δ", solver.Δ, η₁, η₂, σ₁, σ₂)
  @printf(logging, "\n")
  @printf(logging, "| max_eval :   %6d | restol   : %1.2e | res_rtol  : %1.2e | atol : %1.2e | rtol : %1.2e |\n", max_eval, restol, res_rtol, atol, rtol)
  @printf(logging, "| in_rtol  : %1.2e | in_itmax :   %6d | in_conlim : %1.2e |\n\n", in_rtol, in_itmax, in_conlim)
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
  @printf(logging, "| %4s %8s %8s %8s %8s %9s %9s %9s %8s %4s %6s %8s %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "Δ", "Ared", "Pred", "ρ", "Jcond", "sub", "sub-it", "sub-time", "jprod")
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
end

"""
Header of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_header(logging :: IO, model :: AbstractNLSModel, solver :: AbstractLMSolver, η₁ :: AbstractFloat, η₂ :: AbstractFloat, 
                                              σ₁ :: AbstractFloat, σ₂ :: AbstractFloat, max_eval :: Integer, restol :: AbstractFloat, res_rtol :: AbstractFloat, 
                                              atol :: AbstractFloat, rtol :: AbstractFloat, in_rtol :: AbstractFloat, in_itmax :: Integer, in_conlim, :: Val{false})
  @printf(logging, "Solving %s with %d equations and %d variables\n\n", model.meta.name, model.nls_meta.nequ, model.meta.nvar)
  @printf(logging, "Parameters of the solver :\n")
  @printf(logging, "| %1s        : %1.2e | η₁       : %1.2e | η₂        : %1.2e | σ₁   : %1.2e | σ₂   : %1.2e |", "λ", solver.λ, η₁, η₂, σ₁, σ₂)
  @printf(logging, " λmin     : %1.2e |\n", solver.λmin)
  @printf(logging, "| max_eval :   %6d | restol   : %1.2e | res_rtol  : %1.2e | atol : %1.2e | rtol : %1.2e |\n", max_eval, restol, res_rtol, atol, rtol)
  @printf(logging, "| in_rtol  : %1.2e | in_itmax :   %6d | in_conlim : %1.2e |\n\n", in_rtol, in_itmax, in_conlim)
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
  @printf(logging, "| %4s %8s %8s %8s %8s %9s %9s %9s %8s %4s %6s %8s %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "λ", "Ared", "Pred", "ρ", "Jcond", "sub", "sub-it", "sub-time", "jprod")
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
end

"""
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(logging :: IO, model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver, GPUSolver}, iter :: Integer, rNorm :: AbstractFloat, 
                                      ArNorm :: AbstractFloat, dNorm :: AbstractFloat, Ared :: AbstractFloat, Pred :: AbstractFloat, ρ :: AbstractFloat, 
                                      inner_status :: String, step_time :: AbstractFloat, ::Val{true})
  Jcond = solver.in_solver.stats.Acond
  inner_iter = solver.in_solver.stats.niter
  @printf(logging, "| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %1.2e %4s %6d %1.2e %8d |\n", iter, rNorm, ArNorm, dNorm, solver.Δ, Ared, Pred, ρ, 
          Jcond, inner_status, inner_iter, step_time, neval_jprod_residual(model))
end

"""
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(logging :: IO, model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver, GPUSolver}, iter :: Integer, rNorm :: AbstractFloat, 
                                      ArNorm :: AbstractFloat, dNorm :: AbstractFloat, Ared :: AbstractFloat, Pred :: AbstractFloat, ρ :: AbstractFloat, 
                                      inner_status :: String, step_time :: AbstractFloat, ::Val{false})
  Jcond = solver.in_solver.stats.Acond
  inner_iter = solver.in_solver.stats.niter
  @printf(logging, "| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %1.2e %4s %6d %1.2e %8d |\n", iter, rNorm, ArNorm, dNorm, solver.λ, Ared, Pred, ρ, 
          Jcond, inner_status, inner_iter, step_time, neval_jprod_residual(model))
end

#= """
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(logging, iter, rNorm, ArNorm, dNorm, param, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
  @printf(logging, "| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %1.2e %4s %6d %1.2e %8d |\n", iter, rNorm, ArNorm, dNorm, param, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
end =#
