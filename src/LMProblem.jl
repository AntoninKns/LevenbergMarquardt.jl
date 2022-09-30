function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{true})
  generic_solver.Fxm .= generic_solver.Fx
  generic_solver.Fxm .*= -1 
  lsmr!(generic_solver.in_solver, generic_solver.Jx, generic_solver.Fxm, radius = generic_solver.Δ,
        axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
        itmax = in_itmax, conlim = in_conlim)
  return generic_solver.in_solver
end

function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
      in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
      in_conlim :: AbstractFloat, :: Val{false})
  generic_solver.Fxm .= generic_solver.Fx
  generic_solver.Fxm .*= -1 
  lsmr!(generic_solver.in_solver, generic_solver.Jx, generic_solver.Fxm, λ = generic_solver.λ,
        axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
        itmax = in_itmax, conlim = in_conlim)
  return generic_solver.in_solver
end