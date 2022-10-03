"""
Update the parameter Δ of the algorithm in case of bad step
"""
function bad_step_update!(solver :: AbstractLMSolver, σ₁ :: AbstractFloat, ::Val{true})
  solver.Δ = σ₁ * solver.Δ
  return solver.Δ
end

"""
Update the parameter ( λ or Δ ) of the algorithm in case of bad step
"""
function bad_step_update!(solver :: AbstractLMSolver, σ₁ :: AbstractFloat, :: Val{false})
  if solver.λ < solver.λmin
    solver.λ = solver.λmin
  else
    solver.λ = σ₁ * solver.λ
  end
  return solver.λ
end

"""
Update the parameter Δ of the algorithm in case of good step
"""
function good_step_update!(solver :: AbstractLMSolver, T :: Type, :: Val{true})
  return solver.Δ
end

"""
Update the parameter ( λ or Δ ) of the algorithm in case of good step
"""
function good_step_update!(solver :: AbstractLMSolver, T :: Type, :: Val{false})
  if solver.λ < solver.λmin
    solver.λ = zero(T)
  end
  return solver.λ 
end

"""
Update the parameter Δ of the algorithm in case of very good step
"""
function very_good_step_update!(solver :: AbstractLMSolver, σ₂ :: AbstractFloat, :: Val{true})
  solver.Δ = σ₂ * solver.Δ
  return solver.Δ
end

"""
Update the parameter λ of the algorithm in case of very good step
"""
function very_good_step_update!(solver :: AbstractLMSolver, σ₂ :: AbstractFloat, :: Val{false})
  solver.λ = σ₂ * solver.λ
  return solver.λ
end

function rNorm!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})
  return norm(solver.Fx)
end

function rNormp!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})
  return norm(solver.Fxp)
end

function ArNorm!(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver})
  mul!(solver.Jtu, solver.Jx', solver.Fx)
  return norm(solver.Jtu)
end

function ArNorm!(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  mul!(solver.Jtu, solver.A', solver.Fx)
  @views ArNorm = norm(solver.Jtu[m+1:m+n])
  return ArNorm
end

function step!(model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver})
  solver.d .= solver.in_solver.x
  return solver.d
end

function step!(model :: AbstractNLSModel, solver :: Union{MPSolver, GPUSolver, MPGPUSolver})
  copyto!(solver.d, solver.in_solver.x)
  return solver.d
end

function step!(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  @views solver.d = solver.fulld[m+1:m+n]
  return solver.d
end

function ared(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, 
              rNorm :: AbstractFloat, rNormp :: AbstractFloat)
  return rNorm^2 - rNormp^2
end

function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{true})
  mul!(solver.Ju, solver.Jx, solver.d)
  solver.Ju .= solver.Ju .+ solver.Fx
  normJu = norm(solver.Ju)
  return rNorm^2 - (normJu^2 + dNorm^2)
end

function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
  mul!(solver.Ju, solver.Jx, solver.d)
  solver.Ju .= solver.Ju .+ solver.Fx
  normJu = norm(solver.Ju)
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end

function pred(model :: AbstractNLSModel, solver :: LDLSolver, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
  m = model.nls_meta.nequ
  T = eltype(solver.x)
  @views fill!(solver.fulld[1:m], zero(T))
  mul!(solver.Ju, solver.A, solver.fulld)
  @views solver.Ju[1:m] .= solver.Ju[1:m] .+ solver.Fx[1:m]
  @views normJu = norm(solver.Ju[1:m])
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end

function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver, GPUSolver}, 
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  x, d, xp, solver = generic_solver.x, generic_solver.d, generic_solver.xp, generic_solver
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  x .= model.meta.x0
  return x, d, xp, solver
end

function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{MPSolver, MPGPUSolver},
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  solver = generic_solver.F64Solver
  x, d, xp = solver.x, solver.d, solver.xp
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  copyto!(x, model.meta.x0)
  return x, d, xp, solver
end

function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LDLSolver, MINRESSolver},
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  if TR
    error("Impossible to use trust region with LDL factorization")
  end
  x, d, xp, solver = generic_solver.x, generic_solver.d, generic_solver.xp, generic_solver
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  x .= model.meta.x0
  return x, d, xp, solver
end
