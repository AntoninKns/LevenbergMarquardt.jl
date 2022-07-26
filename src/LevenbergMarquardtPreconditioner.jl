export diagonal_precond

function diagonal_precond(model, cols, vals)
  precond = zero(model.meta.nvar)
  for i = 1:model.nls_meta.nnzj
    precond[cols[i]] += vals[i]^2
  end
  return 1.0 ./ precond
end