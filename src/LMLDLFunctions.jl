
function augmented_matrix(model, solver, x, λ)
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
  @views fill!(solver.vals[m+nnzj+1:m+nnzj+n], -λ)

  A = sparse(solver.rows, solver.cols, solver.vals)
  return A
end

function residualLDL!(model, x, Fx)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(x)

  residual!(model, x, view(Fx, 1:m))
  fill!(view(Fx, m+1:m+n), zero(T))
end