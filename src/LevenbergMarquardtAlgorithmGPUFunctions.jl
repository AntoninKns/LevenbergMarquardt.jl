

function residualGPU!(model, x, Fx, GPUFx)
  residual!(model, x, Fx)
  copyto!(GPUFx, Fx)
  return (Fx,GPUFx)
end

"""
Set the first value of the linear operator for the Jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function set_jac_op_residualGPU!(model, solver, x, Jv, Jtv, GPUJv, GPUJtv)
    jac_structure_residual!(model, solver.rows, solver.cols)
    jac_coord_residual!(model, x, solver.vals)
    copyto!(solver.GPUrows, solver.rows)
    copyto!(solver.GPUcols, solver.cols)
    copyto!(solver.GPUvals, solver.vals)
    Jx = sparse(solver.rows, solver.cols, solver.vals)
    GPUJx = CuSparseMatrixCSC(Jx)
  return (Jx,GPUJx)
end

"""
Update the linear operator for the jacobian
Works with sparse manually calculated jacobian or automatic differentiation
"""
function update_jac_op_residualGPU!(model, solver, x, Jv, Jtv, GPUJv, GPUJtv)
  jac_coord_residual!(model, x, solver.vals)
  copyto!(solver.GPUrows, solver.rows)
  copyto!(solver.GPUcols, solver.cols)
  copyto!(solver.GPUvals, solver.vals)
  Jx = sparse(solver.rows, solver.cols, solver.vals)
  GPUJx = CuSparseMatrixCSC(Jx)
  return (Jx,GPUJx)
end
