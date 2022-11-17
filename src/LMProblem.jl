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
  F64Solver.Fxm .= F64Solver.Fx
  F64Solver.Fxm .*= -1 
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

#= """
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: LDLSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
      m = model.nls_meta.nequ
      n = model.meta.nvar
      T = eltype(generic_solver.x)
	generic_solver.Fxm .= generic_solver.Fx
	generic_solver.Fxm .*= -1 
	Au = Symmetric(triu(generic_solver.A), :U)
      D3 = Diagonal(ones(m+n))
      C_k = Diagonal{T}(undef, m+n)
      RipQP.equilibrate!(Au, D3, C_k; ϵ=T(1e-6))
	LDLT = ldl(Au)
      # println(norm(C_k*ones(m+n)))
      D3Fxm = D3*generic_solver.Fxm
      d = similar(generic_solver.fulld)
	ldiv!(d, LDLT, D3Fxm)
      generic_solver.fulld = D3*d
      #ldiv!(generic_solver.fulld, LDLT, generic_solver.Fxm)
      # Suitesparse generic_solver.fulld = LDLT \ generic_solver.Fxm
      #println(norm(generic_solver.fulld))
	return LDLT
end =#

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: LDLSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	m = model.nls_meta.nequ
	n = model.meta.nvar
	T = eltype(generic_solver.x)
	generic_solver.Fxm .= generic_solver.Fx
	generic_solver.Fxm .*= -1 
	LDL = Ma57(generic_solver.A)
	ma57_factorize!(LDL)
#= 	(L, D, s, p) = ma57_get_factors(LDL)
	d1 = diag(D)
	d2 = [diag(D, 1) ; 0][:]
	F = [Vector(d1)' ; Vector(d2)']
	inv_lbl!(F)
	ma57_alter_d(LDL, F) =#
	generic_solver.fulld = ma57_solve(LDL, generic_solver.Fxm)
	return LDL
end

function inv_lbl!(M :: AbstractMatrix)
	n = size(M,2)
	i = 1
	while i <= n
		if abs(M[2,i]) <= 1e-14
			M[1,i] = 1/M[1,i]
			i += 1
		else
			det_M = M[1,i]*M[1,i+1] - M[2,i]^2
			M[1, i], M[1, i+1] = M[1, i+1]/det_M, M[1, i]/det_M
			M[2, i] = -M[2,i]/det_M
			i += 2
		end
	end
	return M
end

function def_pos_lbl(M :: AbstractMatrix)
  n = size(M,2)
	i = 1
	while i <= n
		if abs(M[2,i]) <= 1e-14
			M[1,i] = abs(M[1,i])
			i += 1
		else
			det_M = M[1,i]*M[1,i+1] - M[2,i]^2
			trace_M = M[1,i] + M[1,i+1]
			l1, l2 = Krylov.roots_quadratic(-1., trace_M, -det_M)
			if l1 < 0. || l2 < 0.
				l = max(abs(l1), abs(l2)) + 1.
				M[1, i] += l 
				M[1, i+1] += l
			end
			i += 2
		end
	end
	return M 
end

"""
Solve the sub problem min ‖Jx*d + Fx‖^2 of Levenberg Marquardt Algorithm
Adapting if using a regularized or trust region version
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MINRESSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	generic_solver.Fxm .= generic_solver.Fx
	generic_solver.Fxm .*= -1
  T = eltype(generic_solver.x)
	A32 = Float32.(generic_solver.A)
	LDL = Ma57(Au)
	ma57_factorize!(LDL)
	(L, D, s, p) = ma57_get_factors(LDL)
	println(typeof(D))
	d1 = diag(D)
	println(typeof(d1))
	d2 = [diag(D, 1) ; zero(Float32)][:]
	println(typeof(d2))
	F = [Vector(d1)' ; Vector(d2)']
	println(typeof(F))
	def_pos_lbl(F)
	inv_lbl!(F)
      # D2 = abs.(D)
  ma57_alter_d(LDL, F)
#= 	P = lldl(generic_solver.A)
	P.D .= abs.(P.D) =#
#=       LDL = ldl(generic_solver.A)
	LDL.d .= abs.(LDL.d)
	println(findmax(LDL.d)) =#
#= 	LDL = ldlt(Au)
	LDL.D .= abs.(LDL.D) =#
  # println("minresqlp")
	minres_qlp!(generic_solver.in_solver, generic_solver.A, generic_solver.Fxm,
              M=LDL, itmax=in_itmax, atol=in_atol, rtol=in_rtol, ctol=zero(Float64), ldiv=true)
	return generic_solver.in_solver
end
