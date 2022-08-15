"""
Shortens status log of Levenberg Marquardt subproblem
"""
function change_stats(status::AbstractString)
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
function levenberg_marquardt_log_header(logging, model, TR, param, η₁, η₂, σ₁, σ₂, max_eval, λmin, restol, atol, rtol, in_rtol, in_itmax, in_conlim)
  @printf(logging, "Solving %s with %d equations and %d variables\n\n", model.meta.name, model.nls_meta.nequ, model.meta.nvar)
  @printf(logging, "Parameters of the solver :\n")
  TR ? param_string = "Δ" : param_string = "λ" 
  @printf(logging, "| %1s        : %1.2e | η₁       : %1.2e | η₂        : %1.2e | σ₁   : %1.2e | σ₂   : %1.2e |", param_string, param, η₁, η₂, σ₁, σ₂)
  if TR
    @printf(logging, "\n")
  else
    @printf(logging, " λmin     : %1.2e |\n", λmin)
  end
  @printf(logging, "| max_eval :   %6d | restol   : %1.2e | atol      : %1.2e | rtol : %1.2e |\n", max_eval, restol, atol, rtol)
  @printf(logging, "| in_rtol  : %1.2e | in_itmax :   %6d | in_conlim : %1.2e |\n\n", in_rtol, in_itmax, in_conlim)
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
  @printf(logging, "| %4s %8s %8s %8s %8s %9s %9s %9s %8s %4s %6s %8s %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", param_string, "Ared", "Pred", "ρ", "Jcond", "sub", "sub-it", "sub-time", "jprod")
  @printf(logging, "|---------------------------------------------------------------------------------------------------------------|\n")
end

"""
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(logging, iter, rNorm, ArNorm, dNorm, param, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
  @printf(logging, "| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %1.2e %4s %6d %1.2e %8d |\n", iter, rNorm, ArNorm, dNorm, param, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
end
