
function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, MPSolver})
  residual!(model, x, solver.Fx)
  return solver.Fx
end

function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver, MPGPUSolver})
  residual!(model, x, solver.Fx)
  copyto!(solver.GPUFx, solver.Fx)
  return solver.Fx
end

function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(x)

  residual!(model, x, view(solver.Fx, 1:m))
  fill!(view(solver.Fx, m+1:m+n), zero(T))
  return solver.Fx
end

function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{LMSolver, MPSolver})
  residual!(model, xp, solver.Fxp)
  return solver.Fxp
end

function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{GPUSolver, MPGPUSolver})
  residual!(model, xp, solver.Fxp)
  copyto!(solver.GPUFxp, solver.Fxp)
  return solver.Fxp
end

function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(x)

  residual!(model, xp, view(solver.Fxp, 1:m))
  fill!(view(solver.Fxp, m+1:m+n), zero(T))
  return solver.Fxp
end
