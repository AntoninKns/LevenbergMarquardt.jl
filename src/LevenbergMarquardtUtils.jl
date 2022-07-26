export lm_dot

"""
Function that shortens status log of Levenberg Marquardt subproblem
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
function levenberg_marquardt_log_header(model, λ, η₁, η₂, σ₁, σ₂, max_eval, λmin, restol, atol, rtol, in_rtol, in_itmax, in_conlim)
  @printf("Solving %s with levenberg_marquardt method :\n", model.meta.name)
  @printf("Parameters of the solver :\n")
  @printf("| λ        : %1.2e | η₁       : %1.2e | η₂        : %1.2e | σ₁   : %1.2e | σ₂   : %1.2e |\n", λ, η₁, η₂, σ₁, σ₂)
  @printf("| max_eval :   %6d | λmin     : %1.2e | restol    : %1.2e | atol : %1.2e | rtol : %1.2e |\n", max_eval, λmin, restol, atol, rtol)
  @printf("| in_rtol  : %1.2e | in_itmax :   %6d | in_conlim : %1.2e |\n", in_rtol, in_itmax, in_conlim)
  @printf("|---------------------------------------------------------------------------------------------------------------|\n")
  @printf("| %4s %8s %8s %8s %8s %9s %9s %9s %8s %4s %6s %8s %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "λ", "Ared", "Pred", "ρ", "Jcond", "sub", "sub-it", "sub-time", "jprod")
  @printf("|---------------------------------------------------------------------------------------------------------------|\n")
end

"""
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(iter, rNorm, ArNorm, dNorm, λ, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
  @printf("| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %1.2e %4s %6d %1.2e %8d |\n", iter, rNorm, ArNorm, dNorm, λ, Ared, Pred, ρ, Jcond, inner_status, inner_iter, step_time, jprod)
end

"""
Header of Levenberg Marquardt trust region logs
"""
function levenberg_marquardt_tr_log_header(model)
  @printf("Solving %s with levenberg_marquardt method :\n", model.meta.name)
  @printf("|-------------------------------------------------------------------------------------------|\n")
  @printf("| %4s %8s %8s %8s %8s %9s %9s %9s %4s %6s %6s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "Δ", "Ared", "Pred", "ρ", "sub", "sub-it", "jprod")
  @printf("|-------------------------------------------------------------------------------------------|\n")
end

"""
Row of Levenberg Marquardt trust region logs
"""
function levenberg_marquardt_tr_log_row(iter, rNorm, ArNorm, dNorm, Δ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
  @printf("| %4d %1.2e %1.2e %1.2e %1.2e % 1.2e % 1.2e % 1.2e %4s %6d %6d |\n", iter, rNorm, ArNorm, dNorm, Δ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
end
