"""
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{true})

Solve the sub problem minimize ½‖J(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm.
Using LSMR iterative method and Δ as trust region radius.
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½(‖J(xk) d + F(xk)‖² + λ²‖d‖²) of Levenberg Marquardt Algorithm.
Using LSMR iterative method and λ as regularization parameter.
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	m = model.nls_meta.nequ
	n = model.meta.nvar
  generic_solver.Fxm .= generic_solver.Fx
  generic_solver.Fxm .*= -1
#= 	D1 = Diagonal{Float64}(undef, m)
	D2 = Diagonal{Float64}(undef, n)
	R_k = Diagonal{Float64}(undef, m)
	C_k = Diagonal{Float64}(undef, n)
	A = sparse(generic_solver.rows, generic_solver.cols, generic_solver.vals)
	equilibrate!(A, D1, D2, R_k, C_k, ϵ=10e-6)
	D1Fxm = D1 * generic_solver.Fxm =#
  lsmr!(generic_solver.in_solver, generic_solver.Jx, generic_solver.Fxm, λ = generic_solver.λ,
	# lsmr!(generic_solver.in_solver, A, D1Fxm, λ = generic_solver.λ,
    axtol = in_axtol, btol = in_btol, atol = in_atol, rtol = in_rtol, etol = in_etol,
    itmax = in_itmax, conlim = in_conlim)
	# generic_solver.in_solver.x = D2 * generic_solver.in_solver.x
#= 	c = zeros(n)
	x0 = 1e-2*ones(m)
	y0 = 1e-2*ones(n)
	T = Float64
	if generic_solver.λ < generic_solver.λmin
		generic_solver.λ = generic_solver.λmin
	end
	trimr!(generic_solver.in_solver, generic_solver.Jx, generic_solver.Fxm, c, x0, y0, ν=-generic_solver.λ^2, rtol=in_rtol, atol=zero(T), itmax = in_itmax) =#
  return generic_solver.in_solver
end

"""
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: GPUSolver,
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{true})

Solve the sub problem minimize ½‖J(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm on GPU.
Using LSMR iterative method and Δ as trust region radius.
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: GPUSolver, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½(‖J(xk) d + F(xk)‖² + λ²‖d‖²) of Levenberg Marquardt Algorithm on GPU.
Using LSMR iterative method and λ as regularization parameter.
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPSolver, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{true})

Solve the sub problem minimize ½‖J(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm.
Using iterative refinement on LSMR iterative method and Δ as trust region radius.
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPSolver,
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½(‖J(xk) d + F(xk)‖² + λ²‖d‖²) of Levenberg Marquardt Algorithm.
Using iterative refinement on LSMR iterative method and λ as regularization parameter.
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
		rtol = Float32(1e-2),
		etol = zero(Float32),
		itmax = in_itmax,
		conlim = zero(Float32))
	m = model.nls_meta.nequ
	r1 = Vector{Float64}(undef, m)
	temp = F32Solver.Fxm - F32Solver.Jx * in_solver32.x
	copyto!(r1, temp)
  lsmr!(in_solver64, F64Solver.Jx, r1,
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{true})

Solve the sub problem minimize ½‖J(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm on GPU.
Using iterative refinement on LSMR iterative method and Δ as trust region radius.
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
	copyto!(F32Solver.GPUrows, F64Solver.rows)
	copyto!(F32Solver.GPUcols, F64Solver.cols)
	copyto!(F32Solver.GPUvals, F64Solver.vals)
	F32Solver.Jx = jac_op_residual!(model, F32Solver.rows, F32Solver.cols, F32Solver.vals, F32Solver.Jv, F32Solver.Jtv)
	F32Solver.GPUJx = CuSparseMatrixCOO(F32Solver.GPUrows, F32Solver.GPUcols, F32Solver.GPUvals, (m,n), nnzj)
  lsmr!(in_solver32, F32Solver.GPUJx, F32Solver.GPUFxm,
		radius = F32Solver.Δ,
		axtol = zero(Float32),
		btol = zero(Float32),
		atol = zero(Float32),
		rtol = Float32(1e-2),
		etol = zero(Float32),
		itmax = in_itmax,
		conlim = zero(Float32))
##### NOT WORKING PROPERLY #####
#= copyto!(F64Solver.d, F32Solver.in_solver.x)
F64Solver.x .= F64Solver.x .+ F64Solver.d
residual!(model, F64Solver.x, F64Solver.Fxm)
F64Solver.Fxm .*= -1
copyto!(F64Solver.GPUFxm, F64Solver.Fxm) =#
	copyto!(F32Solver.d, F32Solver.in_solver.x)
	m = model.nls_meta.nequ
	GPUr1 = CuVector{Float64}(undef, m)
	temp = F32Solver.Fxm - F32Solver.Jx * F32Solver.d
	copyto!(GPUr1, temp)
  lsmr!(in_solver64, F64SolverGPU.Jx, GPUr1,
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver,
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½(‖J(xk) d + F(xk)‖² + λ²‖d‖²) of Levenberg Marquardt Algorithm on GPU.
Using iterative refinement on LSMR iterative method and λ as regularization parameter.
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	m = model.nls_meta.nequ
	n = model.meta.nvar
	nnzj = model.nls_meta.nnzj
	F32Solver, F64Solver = generic_solver.F32Solver, generic_solver.F64Solver
	in_solver32, in_solver64 = generic_solver.F32Solver.in_solver, generic_solver.F64Solver.in_solver
	F64Solver.Fxm .= F64Solver.Fx
	F64Solver.Fxm .*= -1 
	copyto!(F32Solver.Fxm, F64Solver.Fxm)
	copyto!(F32Solver.rows, F64Solver.rows)
	copyto!(F32Solver.cols, F64Solver.cols)
	copyto!(F32Solver.vals, F64Solver.vals)
	copyto!(F32Solver.GPUFxm, F64Solver.Fxm)
	copyto!(F32Solver.GPUrows, F64Solver.rows)
	copyto!(F32Solver.GPUcols, F64Solver.cols)
	copyto!(F32Solver.GPUvals, F64Solver.vals)
	F32Solver.Jx = jac_op_residual!(model, F32Solver.rows, F32Solver.cols, F32Solver.vals, F32Solver.Jv, F32Solver.Jtv)
	F32Solver.GPUJx = CuSparseMatrixCOO(F32Solver.GPUrows, F32Solver.GPUcols, F32Solver.GPUvals, (m,n), nnzj)
	lsmr!(in_solver32, F32Solver.GPUJx, F32Solver.GPUFxm,
		λ = F32Solver.λ,
		axtol = zero(Float32),
		btol = zero(Float32),
		atol = zero(Float32),
		rtol = Float32(1e-2),
		etol = zero(Float32),
		itmax = in_itmax,
		conlim = Float32(in_conlim))
	copyto!(F32Solver.d, F32Solver.in_solver.x)
	m = model.nls_meta.nequ
	GPUr1 = CuVector{Float64}(undef, m)
	temp = F32Solver.Fxm - F32Solver.Jx * F32Solver.d
	copyto!(GPUr1, temp)
	lsmr!(in_solver64, F64Solver.GPUJx, GPUr1,
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
generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: LDLSolver,
																							in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																							in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																							in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem [I     J(xₖ)] d = [F(xₖ)] of Levenberg Marquardt Algorithm
											[J(xₖ)ᵀ  -λI]	    [  0  ]
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
#= 	D3 = Diagonal{T}(undef, m+n)
	C_k = Diagonal{T}(undef, m+n)
	equilibrate!(Au, D3, C_k; ϵ=T(1e-6)) =#
	LDLT = ldl(Au)
	# println(norm(C_k*ones(m+n)))
#= 	D3Fxm = D3*generic_solver.Fxm
	d = similar(generic_solver.fulld)
	ldiv!(d, LDLT, D3Fxm)
	generic_solver.fulld = D3*d =#
	ldiv!(generic_solver.fulld, LDLT, generic_solver.Fxm)
	# Suitesparse generic_solver.fulld = LDLT \ generic_solver.Fxm
	#println(norm(generic_solver.fulld))
	return LDLT
end

"""
generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MINRESSolver,
																							in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																							in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																							in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½‖A(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm with LDL factorization preconditioner.
Where A(xk) = [I     J(xₖ)]
							[J(xₖ)ᵀ  -λI]
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: MINRESSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	m = model.nls_meta.nequ
	n = model.meta.nvar
	T = eltype(generic_solver.x)
	generic_solver.Fxm .= generic_solver.Fx
	generic_solver.Fxm .*= -1
#= 	Au = copy(Symmetric(generic_solver.A))
#= 	D3 = Diagonal{T}(undef, m+n)
	C_k = Diagonal{T}(undef, m+n)
	equilibrate!(Au, D3, C_k; ϵ=T(1e-6)) =#
	P = ldl(Au)
	P.D .= abs.(P.D)
	# println(findmax(P.D))
	# D3Fxm = D3*generic_solver.Fxm
	P = lldl(generic_solver.A)
	P.D .= abs.(P.D) =#
  # println("minresqlp")
	# Attention critères d'arrêts de MINRES différents de LSMR, LSQR, etc...
	minres!(generic_solver.in_solver, generic_solver.A, generic_solver.Fxm, rtol=in_rtol, conlim=zero(T), atol=zero(T), etol=zero(T), itmax = in_itmax, ldiv=true)
	# generic_solver.in_solver.x = D3*generic_solver.in_solver.x
	return generic_solver.in_solver
end

"""
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{true})

Solve the sub problem minimize ½‖J(xk) d + F(xk)‖² of Levenberg Marquardt Algorithm.
Using LSMR iterative method and Δ as trust region radius.
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: CGSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
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
		generic_solver.in_solver = solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver}, 
																									in_axtol :: AbstractFloat, in_btol :: AbstractFloat, in_atol :: AbstractFloat, 
																									in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
																									in_conlim :: AbstractFloat, :: Val{false})

Solve the sub problem minimize ½(‖J(xk) d + F(xk)‖² + λ²‖d‖²) of Levenberg Marquardt Algorithm.
Using LSMR iterative method and λ as regularization parameter.
"""
function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: CGSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
                            in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
                            in_conlim :: AbstractFloat, :: Val{false})
	m = model.nls_meta.nequ
	n = model.meta.nvar
  generic_solver.JtFxm .= generic_solver.Jx' * generic_solver.Fx
  generic_solver.JtFxm .*= -1
#= 	D1 = Diagonal{Float64}(undef, m)
	D2 = Diagonal{Float64}(undef, n)
	R_k = Diagonal{Float64}(undef, m)
	C_k = Diagonal{Float64}(undef, n)
	A = sparse(generic_solver.rows, generic_solver.cols, generic_solver.vals)
	equilibrate!(A, D1, D2, R_k, C_k, ϵ=10e-6)
	D1Fxm = D1 * generic_solver.Fxm =#
	# Jx = sparse(generic_solver.rows, generic_solver.cols, generic_solver.vals)
	JtJ = generic_solver.Jx' * generic_solver.Jx
	# println(typeof(JtJ))
  cg!(generic_solver.in_solver, JtJ, generic_solver.JtFxm,
	# lsmr!(generic_solver.in_solver, A, D1Fxm, λ = generic_solver.λ,
    atol = in_atol, rtol = in_rtol, itmax = in_itmax)
	# generic_solver.in_solver.x = D2 * generic_solver.in_solver.x
  return generic_solver.in_solver
end

function solve_sub_problem!(model :: AbstractNLSModel, generic_solver :: SCHURSolver, in_axtol :: AbstractFloat, in_btol :: AbstractFloat, 
														in_atol :: AbstractFloat, in_rtol :: AbstractFloat, in_etol :: AbstractFloat, in_itmax :: Integer, 
														in_conlim :: AbstractFloat, :: Val{false})
	npnts = model.npnts
	ncams = model.ncams
	mul!(generic_solver.JtFxm, generic_solver.Jx', generic_solver.Fx)
	generic_solver.JtFxm .*= -1
	generic_solver.JtJ = generic_solver.Jx'*generic_solver.Jx
	generic_solver.JtJ .= generic_solver.JtJ + generic_solver.λ * I
	generic_solver.B .= generic_solver.JtJ[1:3*npnts,1:3*npnts]
	inv_block_diag!(generic_solver.B)
	@views generic_solver.C = generic_solver.JtJ[3*npnts+1:end, 3*npnts+1:end]
	@views generic_solver.E = generic_solver.JtJ[1:3*npnts, 3*npnts+1:end]
	generic_solver.Schur .= generic_solver.C .- generic_solver.E'* generic_solver.B * generic_solver.E
	LDLT = ldl(generic_solver.Schur)
	u = generic_solver.JtFxm[3*npnts+1:end] - generic_solver.E'*generic_solver.B*generic_solver.JtFxm[1:3*npnts]
	@views ldiv!(generic_solver.d[3*npnts+1:end], LDLT, u)
	generic_solver.d[1:3*npnts] = generic_solver.B*(generic_solver.JtFxm[1:3*npnts]-generic_solver.E*generic_solver.d[3*npnts+1:end])
	return LDLT
end