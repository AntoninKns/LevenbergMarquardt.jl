
"""
    solver.Jx = set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LMSolver)

Set the first value of the linear operator for the Jacobian.
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LMSolver)
  jac_structure_residual!(model, solver.rows, solver.cols)
  jac_coord_residual!(model, x, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  return solver.Jx
end


"""
    solver.Jx = set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)

Set the first value of the linear operator for the Jacobian using automatic differentiation.
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)
  solver.Jx = jac_op_residual!(model, x, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
    solver.Jx = set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: GPUSolver)

Set the first value of the Jacobian for CPU and GPU calculations. 
For the sake of code genericity only `solver.Jx` is returned but `solver.GPUJx` is also calculated.
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: GPUSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  jac_structure_residual!(model, solver.rows, solver.cols)
  jac_coord_residual!(model, x, solver.vals)
  copyto!(solver.GPUrows, solver.rows)
  copyto!(solver.GPUcols, solver.cols)
  copyto!(solver.GPUvals, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  solver.GPUJx = CuSparseMatrixCOO(solver.GPUrows, solver.GPUcols, solver.GPUvals, (m,n), nnzj)
  return solver.Jx
end

"""
    solver.A = set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LDLSolver)

Set the first value of the augmented matrix for LDL factorization.

The augmented matrix should have the following structure:

[I     J]
[J^T -λI]

In this function we only calculate:

[I   J]
[0 -λI]

We then use a `Symmetric` wrapper to make it symmetric.
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  T = eltype(x)

  @views jac_rows = solver.rows[m+1:m+nnzj]
  @views jac_cols = solver.cols[m+1:m+nnzj]
  @views jac_vals = solver.vals[m+1:m+nnzj]

  jac_structure_residual!(model, jac_rows, jac_cols)
  jac_coord_residual!(model, x, jac_vals)

  jac_cols .+= m

  @views solver.rows[1:m] = 1:m
  @views solver.cols[1:m] = 1:m
  @views fill!(solver.vals[1:m], T(1.))

  @views solver.rows[m+nnzj+1:m+nnzj+n] = m+1:m+n
  @views solver.cols[m+nnzj+1:m+nnzj+n] = m+1:m+n
  @views fill!(solver.vals[m+nnzj+1:m+nnzj+n], -solver.λ)

  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

"""
    solver.A = set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: MINRESSolver)

Set the first value of the augmented matrix for MINRES resolution.

The augmented matrix has the following structure:
  
[I     J]
[J^T -λI]
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  T = eltype(x)

  @views jac_rows = solver.rows[m+1:m+nnzj]
  @views jac_cols = solver.cols[m+1:m+nnzj]
  @views jac_vals = solver.vals[m+1:m+nnzj]

  jac_structure_residual!(model, jac_rows, jac_cols)
  jac_coord_residual!(model, x, jac_vals)

  jac_cols .+= m

  @views copyto!(solver.rows[m+nnzj+1:m+2*nnzj], solver.cols[m+1:m+nnzj])
  @views copyto!(solver.cols[m+nnzj+1:m+2*nnzj], solver.rows[m+1:m+nnzj])
  @views copyto!(solver.vals[m+nnzj+1:m+2*nnzj], solver.vals[m+1:m+nnzj])

  @views solver.rows[1:m] = 1:m
  @views solver.cols[1:m] = 1:m
  @views fill!(solver.vals[1:m], T(1.))

  @views solver.rows[m+2*nnzj+1:m+2*nnzj+n] = m+1:m+n
  @views solver.cols[m+2*nnzj+1:m+2*nnzj+n] = m+1:m+n
  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -solver.λ)

  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

"""
    solver.Jx = update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LMSolver)

Update the linear operator for the Jacobian.
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LMSolver)
  jac_coord_residual!(model, x, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
    solver.Jx = update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)

Update the linear operator for the Jacobian using automatic differentiation.
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)
  solver.Jx = jac_op_residual!(model, x, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
    solver.Jx = update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: GPUSolver)

Update the Jacobian for CPU and GPU calculations. 
For the sake of code genericity only `solver.Jx` is returned but `solver.GPUJx` is also calculated.
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: GPUSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  jac_coord_residual!(model, x, solver.vals)
  copyto!(solver.GPUrows, solver.rows)
  copyto!(solver.GPUcols, solver.cols)
  copyto!(solver.GPUvals, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  solver.GPUJx = CuSparseMatrixCOO(solver.GPUrows, solver.GPUcols, solver.GPUvals, (m,n), nnzj)
  return solver.Jx
end

"""
    solver.A = update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LDLSolver)

Update the augmented matrix for LDL factorization.

The augmented matrix should have the following structure:

[I     J]
[J^T -λI]

In this function we only calculate:

[I   J]
[0 -λI]

We then use a `Symmetric` wrapper to make it symmetric.
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  T = eltype(x)

  @views jac_vals = solver.vals[m+1:m+nnzj]

  jac_coord_residual!(model, x, jac_vals)

  @views solver.rows[1:m] = 1:m
  @views solver.cols[1:m] = 1:m
  @views fill!(solver.vals[1:m], T(1.))

  @views solver.rows[m+nnzj+1:m+nnzj+n] = m+1:m+n
  @views solver.cols[m+nnzj+1:m+nnzj+n] = m+1:m+n
  @views fill!(solver.vals[m+nnzj+1:m+nnzj+n], -solver.λ)

  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

"""
    solver.A = update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: MINRESSolver)
    
Update the augmented matrix for MINRES resolution.

The augmented matrix has the following structure:
  
[I     J]
[J^T -λI]
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj
  T = eltype(x)

  @views jac_vals = solver.vals[m+1:m+nnzj]

  jac_coord_residual!(model, x, jac_vals)

  @views copyto!(solver.rows[m+nnzj+1:m+2*nnzj], solver.cols[m+1:m+nnzj])
  @views copyto!(solver.cols[m+nnzj+1:m+2*nnzj], solver.rows[m+1:m+nnzj])
  @views copyto!(solver.vals[m+nnzj+1:m+2*nnzj], solver.vals[m+1:m+nnzj])

  @views solver.rows[1:m] = 1:m
  @views solver.cols[1:m] = 1:m
  @views fill!(solver.vals[1:m], T(1.))

  @views solver.rows[m+2*nnzj+1:m+2*nnzj+n] = m+1:m+n
  @views solver.cols[m+2*nnzj+1:m+2*nnzj+n] = m+1:m+n
  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -solver.λ)

  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

"""
    solver.Jx = update_lambda!(model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver, GPUSolver})

Update λ in case of a good or very good step.
"""
function update_lambda!(model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver, GPUSolver})
  # For these solvers, λ is updated directly in the subproblem, this function exists only
  # for better code genericity
  return solver.Jx
end

"""
    solver.A = update_lambda!(model :: AbstractNLSModel, solver :: LDLSolver)

Update λ in the augmented matrix for LDL factorization.

The augmented matrix should have the following structure:

[I     J]
[J^T -λI]

In this function we only calculate:

[I   J]
[0 -λI]

We then use a `Symmetric` wrapper to make it symmetric.
"""
function update_lambda!(model :: AbstractNLSModel, solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  @views fill!(solver.vals[m+nnzj+1:m+nnzj+n], -solver.λ)
  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

"""
    solver.A = update_lambda!(model :: AbstractNLSModel, solver :: MINRESSolver)
    
Update λ in the augmented matrix for MINRES resolution.

The augmented matrix has the following structure:
  
[I     J]
[J^T -λI]
"""
function update_lambda!(model :: AbstractNLSModel, solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -solver.λ)
  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

#= function NLPModels.jprod_residual!(nls :: AbstractNLSModel,
                                   rows :: AbstractVector{<:Integer},
                                   cols :: AbstractVector{<:Integer},
                                   vals :: Union{Vector{Float32}, Vector{Float64}},
                                   Jv :: AbstractVector)
  @lencheck nls.nls_meta.nnzj rows cols vals
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
=#