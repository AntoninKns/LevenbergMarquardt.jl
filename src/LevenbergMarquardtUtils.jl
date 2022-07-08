export lm_dot

"""
Function that shortens status log of Levenberg Marquardt subproblem
"""
function change_stats(status::AbstractString)
  if status == "maximum number of iterations exceeded"
    status = "max iter"
  elseif status == "condition number seems too large for this machine"
    status = "cond num"
  elseif status == "condition number exceeds tolerance"
    status = "cond num"
  elseif status == "found approximate minimum least-squares solution"
    status = "solved"
  elseif status == "found approximate zero-residual solution"
    status = "zero res"
  elseif status == "truncated forward error small enough"
    status = "fwd err"
  elseif status == "on trust-region boundary"
    status = "TR bound"
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
function levenberg_marquardt_log_header(model)
  @printf("Solving %s with levenberg_marquardt method :\n", model.meta.name)
  @printf("|------------------------------------------------------------------------------------------------------------------------------|\n")
  @printf("| %4s | %9s | %8s | %8s | %8s | %9s | %9s | %9s | %12s | %10s | %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "λ", "Ared", "Pred", "ρ", "inner status", "inner iter", "jprod")
  @printf("|------|-----------|----------|----------|----------|-----------|-----------|-----------|--------------|------------|----------|\n")
end

"""
Row of Levenberg Marquardt logs
"""
function levenberg_marquardt_log_row(iter, rNorm, ArNorm, dNorm, λ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
  @printf("| %4d | %1.3e | %1.2e | %1.2e | %1.2e | % 1.2e | % 1.2e | % 1.2e | %12s | %10d | %8d |\n", iter, rNorm, ArNorm, dNorm, λ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
end

"""
Header of Levenberg Marquardt trust region logs
"""
function levenberg_marquardt_tr_log_header(model)
  @printf("Solving %s with levenberg_marquardt method :\n", model.meta.name)
  @printf("|------------------------------------------------------------------------------------------------------------------------------|\n")
  @printf("| %4s | %9s | %8s | %8s | %8s | %9s | %9s | %9s | %12s | %10s | %8s |\n", "iter", "‖F(x)‖", "‖J'F‖", "‖d‖", "Δ", "Ared", "Pred", "ρ", "inner status", "inner iter", "jprod")
  @printf("|------|-----------|----------|----------|----------|-----------|-----------|-----------|--------------|------------|----------|\n")
end

"""
Row of Levenberg Marquardt trust region logs
"""
function levenberg_marquardt_tr_log_row(iter, rNorm, ArNorm, dNorm, Δ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
  @printf("| %4d | %1.3e | %1.2e | %1.2e | %1.2e | % 1.2e | % 1.2e | % 1.2e | %12s | %10d | %8d |\n", iter, rNorm, ArNorm, dNorm, Δ, Ared, Pred, ρ, inner_status, inner_iter, jprod)
end
