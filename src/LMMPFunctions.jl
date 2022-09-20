"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model, meta_solver, Jx, Fxm, TR, param,
                            in_axtol, in_btol, in_atol, in_rtol,
                            in_etol, in_itmax, in_conlim)
  F32Solver, F64Solver = meta_solver.F32Solver, meta_solver.F64Solver
  in_solver32, in_solver64 = meta_solver.F32Solver.in_solver, meta_solver.F64Solver.in_solver
  copyto!(F32Solver.Fxm, Fxm)
  copyto!(F32Solver.rows, F64Solver.rows)
  copyto!(F32Solver.cols, F64Solver.cols)
  copyto!(F32Solver.vals, F64Solver.vals)
  Jx32 = jac_op_residual!(model, F32Solver.rows, F32Solver.cols, F32Solver.vals, F32Solver.Jv, F32Solver.Jtv)
  in_solver32 = lsmr!(in_solver32, Jx32, F32Solver.Fxm,
                      radius = zero(Float32),
                      axtol = zero(Float32),
                      btol = zero(Float32),
                      atol = zero(Float32),
                      rtol = Float32(sqrt(eps(Float32))),
                      etol = zero(Float32),
                      itmax = in_itmax,
                      conlim = Float32(in_conlim))
  d = Vector{Float64}(in_solver32.x)
  F64Solver.x .= F64Solver.x .+ d
  Fx2 = similar(Fxm)
  residual!(model, F64Solver.x, Fx2)
  Fxm .= Fx2
  Fxm .*= -1
  in_solver64 = lsmr!(in_solver64, Jx, Fxm,
                      radius = param,
                      axtol = in_axtol,
                      btol = in_btol,
                      atol = in_atol,
                      rtol = in_rtol,
                      etol = in_etol,
                      itmax = in_itmax,
                      conlim = in_conlim)
  return in_solver64
end
