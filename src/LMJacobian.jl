
"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, MPSolver})
  jac_structure_residual!(model, solver.rows, solver.cols)
  jac_coord_residual!(model, x, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  return solver.Jx
end


"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)
  solver.Jx = jac_op_residual!(model, x, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function set_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver, MPGPUSolver})
  jac_structure_residual!(model, solver.rows, solver.cols)
  jac_coord_residual!(model, x, solver.vals)
  copyto!(solver.GPUrows, solver.rows)
  copyto!(solver.GPUcols, solver.cols)
  copyto!(solver.GPUvals, solver.vals)
  solver.Jx = sparse(solver.rows, solver.cols, solver.vals)
  solver.GPUJx = CuSparseMatrixCSC(solver.Jx)
  return solver.Jx
end

"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
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
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
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
Update the linear operator for the jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, MPSolver})
  jac_coord_residual!(model, x, solver.vals)
  solver.Jx = jac_op_residual!(model, solver.rows, solver.cols, solver.vals, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
Update the linear operator for the jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: ADSolver)
  solver.Jx = jac_op_residual!(model, x, solver.Jv, solver.Jtv)
  return solver.Jx
end

"""
Update the linear operator for the jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function update_jac_residual!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver, MPGPUSolver})
  jac_coord_residual!(model, x, solver.vals)
  copyto!(solver.GPUrows, solver.rows)
  copyto!(solver.GPUcols, solver.cols)
  copyto!(solver.GPUvals, solver.vals)
  solver.Jx = sparse(solver.rows, solver.cols, solver.vals)
  solver.GPUJx = CuSparseMatrixCSC(solver.Jx)
  return solver.Jx
end

"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
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
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
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

function update_lambda!(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver})
  return solver.Jx
end

function update_lambda!(model :: AbstractNLSModel, solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  @views fill!(solver.vals[m+nnzj+1:m+nnzj+n], -solver.λ)
  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end

function update_lambda!(model :: AbstractNLSModel, solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -solver.λ)
  solver.A = sparse(solver.rows, solver.cols, solver.vals)
  return solver.A
end
