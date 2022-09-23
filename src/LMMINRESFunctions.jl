function augmented_matrix_MINRES(model, solver, x, λ)
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
  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -λ)

  A = sparse(solver.rows, solver.cols, solver.vals)
  return A
end

function update_jacobian_MINRES(model, solver, λ)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  nnzj = model.nls_meta.nnzj

  @views fill!(solver.vals[m+2*nnzj+1:m+2*nnzj+n], -λ)
  A = sparse(solver.rows, solver.cols, solver.vals)
  return A
end

function residualMINRES!(model, x, Fx)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(x)

  residual!(model, x, view(Fx, 1:m))
  fill!(view(Fx, m+1:m+n), zero(T))
end