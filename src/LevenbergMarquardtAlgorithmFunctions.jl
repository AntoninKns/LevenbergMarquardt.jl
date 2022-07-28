"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(in_solver, Jx, Fxm, TR, param,
                            in_axtol, in_btol, in_atol, in_rtol,
                            in_etol, in_itmax, in_conlim)
  if TR
    in_solver = lsmr!(in_solver, Jx, Fxm,
                      radius = param,
                      axtol = in_axtol,
                      btol = in_btol,
                      atol = in_atol,
                      rtol = in_rtol,
                      etol = in_etol,
                      itmax = in_itmax,
                      conlim = in_conlim)
  else
    in_solver = lsmr!(in_solver, Jx, Fxm,
                      λ = param,
                      axtol = in_axtol,
                      btol = in_btol,
                      atol = in_atol,
                      rtol = in_rtol,
                      etol = in_etol,
                      itmax = in_itmax,
                      conlim = in_conlim)
  end
  return in_solver
end

"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function set_jac_op_residual!(model, solver, T, S, x, Jv, Jtv)
  if typeof(solver) == LMSolver{T, S}
    jac_structure_residual!(model, solver.rows, solver.cols)
    jac_coord_residual!(model, x, solver.vals)
    Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, Jv, Jtv)
  else
    Jx = jac_op_residual!(model, x, Jv, Jtv)
  end
  return Jx
end

"""
Update the linear operator for the jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function update_jac_op_residual!(model, solver, T, S, x, Jv, Jtv)
  if typeof(solver) == LMSolver{T,S}
    jac_coord_residual!(model, x, solver.vals)
    Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, Jv, Jtv)
  else
    Jx = jac_op_residual!(model, x, Jv, Jtv)
  end
  return Jx
end

"""
Update the parameter ( λ or Δ ) of the algorithm in case of bad step
"""
function bad_step_update!(param, TR, σ₁, λmin)
  if TR
    param = σ₁ * param
  else
    if param < λmin
      param = λmin
    else
      param = σ₁ * param
    end
  end
  return param
end

"""
Update the parameter ( λ or Δ ) of the algorithm in case of good step
"""
function good_step_update!(param, TR, λmin, T)
  if !TR
    if param < λmin
      param = zero(T)
    end
  end
  return param
end

"""
Update the parameter ( λ or Δ ) of the algorithm in case of very good step
"""
function very_good_step_update!(param, σ₂)
  param = σ₂ * param
  return param
end