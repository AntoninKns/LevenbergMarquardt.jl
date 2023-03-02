"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, ADSolver})

Calculate F(xk) = ½‖f(xk)‖².
"""
function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, ADSolver, CGSolver, SCHURSolver})
  residual!(model, x, solver.Fx)
  return solver.Fx
end

"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver})

Calculate F(xk) = ½‖f(xk)‖² and copy it to GPU vector.
"""
function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver})
  residual!(model, x, solver.Fx)
  copyto!(solver.GPUFx, solver.Fx)
  return solver.Fx
end

"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, ADSolver})

Calculate F(xk) = ½‖f(xk)‖² and then create vector [F(xk)].
                                                   [  0  ]
"""
function residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(x)

  residual!(model, x, view(solver.Fx, 1:m))
  fill!(view(solver.Fx, m+1:m+n), zero(T))
  return solver.Fx
end

"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, ADSolver})

Calculate F(xk+1) = ½‖f(xk+1)‖².
"""
function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{LMSolver, ADSolver, CGSolver, SCHURSolver})
  residual!(model, xp, solver.Fxp)
  return solver.Fxp
end

"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{GPUSolver})

Calculate F(xk+1) = ½‖f(xk+1)‖² and copy it to GPU vector.
"""
function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{GPUSolver})
  residual!(model, xp, solver.Fxp)
  copyto!(solver.GPUFxp, solver.Fxp)
  return solver.Fxp
end

"""
    solver.Fx = residualLM!(model :: AbstractNLSModel, x :: AbstractVector, solver :: Union{LMSolver, ADSolver})

Calculate F(xk+1) = ½‖f(xk+1)‖² and then create vector [F(xk+1)].
                                                       [  0    ]
"""
function residualLMp!(model :: AbstractNLSModel, xp :: AbstractVector, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(xp)

  residual!(model, xp, view(solver.Fxp, 1:m))
  fill!(view(solver.Fxp, m+1:m+n), zero(T))
  return solver.Fxp
end
