"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
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

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
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

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: GPUSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{true})
  generic_solver.GPUFxm .= generic_solver.GPUFx
  generic_solver.GPUFxm .*= -1 
  lsmr!(generic_solver.in_solver, generic_solver.GPUJx, generic_solver.GPUFxm, radius = generic_solver.Δ,
        axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
        itmax = in_itmax, conlim = in_conlim)
  return generic_solver.in_solver
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: GPUSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	generic_solver.GPUFxm .= generic_solver.GPUFx
	generic_solver.GPUFxm .*= -1
  lsmr!(generic_solver.in_solver, generic_solver.GPUJx, generic_solver.GPUFxm, λ = generic_solver.λ,
        axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
        itmax = in_itmax, conlim = in_conlim)
  return generic_solver.in_solver
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{true})
  F32Solver, F64Solver = generic_solver.F32Solver, generic_solver.F64Solver
  in_solver32, in_solver64 = generic_solver.F32Solver.in_solver, generic_solver.F64Solver.in_solver
  F64Solver.Fxm .= F64Solver.Fx
  F64Solver.Fxm .*= -1 
  copyto!(F32Solver.Fxm, F64Solver.Fxm)
  copyto!(F32Solver.rows, F64Solver.rows)
  copyto!(F32Solver.cols, F64Solver.cols)
  copyto!(F32Solver.vals, F64Solver.vals)
  F32Solver.Jx = jac_op_residual!(model, F32Solver.rows, F32Solver.cols, F32Solver.vals, F32Solver.Jv, F32Solver.Jtv)
  lsmr!(in_solver32, F32Solver.Jx, F32Solver.Fxm,
        radius = F32Solver.Δ,
        axtol = zero(Float32),
        btol = zero(Float32),
        atol = zero(Float32),
        rtol = Float32(sqrt(eps(Float32))),
        etol = zero(Float32),
        itmax = in_itmax,
        conlim = Float32(in_conlim))
  copyto!(F64Solver.d, F32Solver.in_solver.x)
  F64Solver.x .= F64Solver.x .+ F64Solver.d
  residual!(model, F64Solver.x, F64Solver.Fxm)
  F64Solver.Fxm .*= -1
  lsmr!(in_solver64, F64Solver.Jx, F64Solver.Fxm,
				radius = F64Solver.Δ,
				axtol = in_axtol,
				btol = in_btol,
				atol = in_atol,
				rtol = in_rtol,
				etol = in_etol,
				itmax = in_itmax,
				conlim = in_conlim)
  return in_solver64
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
  F32Solver, F64Solver = generic_solver.F32Solver, generic_solver.F64Solver
  in_solver32, in_solver64 = generic_solver.F32Solver.in_solver, generic_solver.F64Solver.in_solver
  generic_solver.Fxm .= generic_solver.Fx
  generic_solver.Fxm .*= -1 
  copyto!(F32Solver.Fxm, F64Solver.Fxm)
  copyto!(F32Solver.rows, F64Solver.rows)
  copyto!(F32Solver.cols, F64Solver.cols)
  copyto!(F32Solver.vals, F64Solver.vals)
  F32Solver.Jx = jac_op_residual!(model, F32Solver.rows, F32Solver.cols, F32Solver.vals, F32Solver.Jv, F32Solver.Jtv)
  lsmr!(in_solver32, F32Solver.Jx, F32Solver.Fxm,
        λ = F32Solver.λ,
        axtol = zero(Float32),
        btol = zero(Float32),
        atol = zero(Float32),
        rtol = Float32(sqrt(eps(Float32))),
        etol = zero(Float32),
        itmax = in_itmax,
        conlim = Float32(in_conlim))
  copyto!(F64Solver.d, F32Solver.in_solver.x)
  F64Solver.x .= F64Solver.x .+ F64Solver.d
  residual!(model, F64Solver.x, F64Solver.Fxm)
  F64Solver.Fxm .*= -1
  lsmr!(in_solver64, F64Solver.Jx, F64Solver.Fxm,
				λ = F64Solver.λ,
				axtol = in_axtol,
				btol = in_btol,
				atol = in_atol,
				rtol = in_rtol,
				etol = in_etol,
				itmax = in_itmax,
				conlim = in_conlim)
  return in_solver64
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{true})
  F32Solver, F64Solver = generic_solver.F32Solver, generic_solver.F64Solver
  in_solver32, in_solver64 = generic_solver.F32Solver.in_solver, generic_solver.F64Solver.in_solver
  F64Solver.Fxm .= F64Solver.Fx
  F64Solver.Fxm .*= -1 
  copyto!(F32Solver.Fxm, F64Solver.Fxm)
  copyto!(F32Solver.rows, F64Solver.rows)
  copyto!(F32Solver.cols, F64Solver.cols)
  copyto!(F32Solver.vals, F64Solver.vals)
  copyto!(F32Solver.GPUFxm, F64Solver.Fxm)
  F32Solver.Jx = sparse(F32Solver.rows, F32Solver.cols, F32Solver.vals)
	F32Solver.GPUJx = CuSparseMatrixCSC(F32Solver.Jx)
  lsmr!(in_solver32, F32Solver.GPUJx, F32Solver.GPUFxm,
        radius = F32Solver.Δ,
        axtol = zero(Float32),
        btol = zero(Float32),
        atol = zero(Float32),
        rtol = Float32(sqrt(eps(Float32))),
        etol = zero(Float32),
        itmax = in_itmax,
        conlim = Float32(in_conlim))
  copyto!(F64Solver.d, F32Solver.in_solver.x)
  F64Solver.x .= F64Solver.x .+ F64Solver.d
  residual!(model, F64Solver.x, F64Solver.Fxm)
  F64Solver.Fxm .*= -1
	copyto!(F64Solver.GPUFxm, F64Solver.Fxm)
  lsmr!(in_solver64, F64SolverGPU.Jx, F64Solver.GPUFxm,
				radius = F64Solver.Δ,
				axtol = in_axtol,
				btol = in_btol,
				atol = in_atol,
				rtol = in_rtol,
				etol = in_etol,
				itmax = in_itmax,
				conlim = in_conlim)
  return in_solver64
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	F32Solver, F64Solver = generic_solver.F32Solver, generic_solver.F64Solver
	in_solver32, in_solver64 = generic_solver.F32Solver.in_solver, generic_solver.F64Solver.in_solver
	F64Solver.Fxm .= F64Solver.Fx
	F64Solver.Fxm .*= -1 
	copyto!(F32Solver.Fxm, F64Solver.Fxm)
	copyto!(F32Solver.rows, F64Solver.rows)
	copyto!(F32Solver.cols, F64Solver.cols)
	copyto!(F32Solver.vals, F64Solver.vals)
	copyto!(F32Solver.GPUFxm, F64Solver.Fxm)
	F32Solver.Jx = sparse(F32Solver.rows, F32Solver.cols, F32Solver.vals)
	F32Solver.GPUJx = CuSparseMatrixCSC(F32Solver.Jx)
	lsmr!(in_solver32, F32Solver.GPUJx, F32Solver.GPUFxm,
				λ = F32Solver.λ,
				axtol = zero(Float32),
				btol = zero(Float32),
				atol = zero(Float32),
				rtol = Float32(sqrt(eps(Float32))),
				etol = zero(Float32),
				itmax = in_itmax,
				conlim = Float32(in_conlim))
	copyto!(F64Solver.d, F32Solver.in_solver.x)
	F64Solver.x .= F64Solver.x .+ F64Solver.d
	residual!(model, F64Solver.x, F64Solver.Fxm)
	F64Solver.Fxm .*= -1
	copyto!(F64Solver.GPUFxm, F64Solver.Fxm)
	lsmr!(in_solver64, F64Solver.GPUJx, F64Solver.GPUFxm,
				λ = F64Solver.λ,
				axtol = in_axtol,
				btol = in_btol,
				atol = in_atol,
				rtol = in_rtol,
				etol = in_etol,
				itmax = in_itmax,
				conlim = in_conlim)
	return in_solver64
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: LDLSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	generic_solver.Fxm .= generic_solver.Fx
	generic_solver.Fxm .*= -1 
	Au = Symmetric(triu(generic_solver.A), :U)
	LDLT = ldl(Au)
	ldiv!(generic_solver.fulld, LDLT, generic_solver.Fxm)
	return generic_solver.fulld
end
