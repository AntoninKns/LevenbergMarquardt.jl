function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: LMSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: AbstractFloat, 
                            in_conlim :: AbstractFloat, :: Val{true})
  lsmr!(generic_solver.in_solver, generic_solver.Jx, generic_solver.Fxm, radius = generic_solver.TR,
        axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
        itmax = in_itmax, conlim = in_conlim)
  return generic_solver.in_solver
end