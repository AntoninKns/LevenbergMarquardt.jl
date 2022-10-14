"""
    solver.Δ = bad_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, σ₁ :: AbstractFloat, ::Val{true})

Update the parameter Δ of the algorithm in the case of a bad step.
"""
function bad_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, σ₁ :: AbstractFloat, ::Val{true})
  solver.Δ = σ₁ * solver.Δ
  return solver.Δ
end

"""
    solver.λ = bad_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, σ₁ :: AbstractFloat, :: Val{false})

Update the parameter λ of the algorithm in the case of a bad step.
"""
function bad_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, σ₁ :: AbstractFloat, :: Val{false})
  # If λ == 0. then we set it to λmin. `<` symbol is used to avoid `==` comparison between float numbers.
  if solver.λ < solver.λmin
    solver.λ = solver.λmin
  else
    solver.λ = σ₁ * solver.λ
  end
  return solver.λ
end

"""
    solver.λ = bad_step_update!(solver :: Union{LDLSolver, MINRESSolver}, σ₁ :: AbstractFloat, :: Val{false})

Update the parameter λ of the algorithm in the case of a bad step.
"""
function bad_step_update!(solver :: Union{LDLSolver, MINRESSolver}, σ₁ :: AbstractFloat, :: Val{false})
  solver.λ = σ₁ * solver.λ
  return solver.λ
end

"""
    solver.Δ = good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, T :: Type, :: Val{true})

Update the parameter Δ of the algorithm in the case of a good step.
"""
function good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, T :: Type, :: Val{true})
  # In case of a good step we don't change Δ. This function purpose is to have a more generic and performant algorithm using multiple dispatch.
  return solver.Δ
end

"""
    solver.λ = good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, T :: Type, :: Val{false})

Update the parameter Δ of the algorithm in the case of a good step.
"""
function good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, T :: Type, :: Val{false})
  # If λ == 0. then we set it to 0. `<` symbol is used to avoid `==` comparison between float numbers.
  if solver.λ < solver.λmin
    solver.λ = zero(T)
  end
  return solver.λ 
end

"""
    solver.λ = good_step_update!(solver :: Union{LDLSolver, MINRESSolver}, T :: Type, :: Val{false})

Update the parameter Δ of the algorithm in the case of a good step.
"""
function good_step_update!(solver :: Union{LDLSolver, MINRESSolver}, T :: Type, :: Val{false})
  return solver.λ 
end

"""
    solver.Δ = very_good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{true})

Update the parameter Δ of the algorithm in the case of a very good step.
"""
function very_good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{true})
  solver.Δ = σ₂ * solver.Δ
  return solver.Δ
end

"""
    solver.λ = very_good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{false})

Update the parameter Δ of the algorithm in the case of a very good step.
"""
function very_good_step_update!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{false})
  solver.λ = σ₂ * solver.λ
  return solver.λ
end

"""
    solver.λ = very_good_step_update!(solver :: Union{LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{false})

Update the parameter Δ of the algorithm in the case of a very good step.
"""
function very_good_step_update!(solver :: Union{LDLSolver, MINRESSolver}, σ₂ :: AbstractFloat, :: Val{false})
  solver.λ = σ₂ * solver.λ
  # If λ == 0. then we set it to λmin. `<` symbol is used to avoid `==` comparison between float numbers.
  if solver.λ < solver.λmin
    solver.λ = solver.λmin
  end
  return solver.λ
end

"""
    rNorm!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})

Calculate ∥F(xₖ)∥.
"""
function rNorm!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})
  return norm(solver.Fx)
end

"""
    rNormp!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})

Calculate ∥F(x_{k+1})∥.
"""
function rNormp!(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver})
  return norm(solver.Fxp)
end


"""
    ArNorm!(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver})

Calculate ∥J(xₖ) × F(xₖ)∥.
"""
function ArNorm!(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver})
  mul!(solver.Jtu, solver.Jx', solver.Fx)
  return norm(solver.Jtu)
end

"""
    ArNorm!(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver})

Calculate ∥J(xₖ) × F(xₖ)∥.
"""
function ArNorm!(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver})
  m = model.nls_meta.nequ
  n = model.meta.nvar
  mul!(solver.Jtu, solver.A', solver.Fx)
  @views ArNorm = norm(solver.Jtu[m+1:m+n])
  return ArNorm
end

"""

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, solver :: Union{LMSolver, ADSolver})
  solver.d .= solver.in_solver.x
  return solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: Union{MPSolver, GPUSolver, MPGPUSolver})

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, solver :: Union{MPSolver, GPUSolver, MPGPUSolver})
  copyto!(solver.d, solver.in_solver.x)
  return solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: LDLSolver)

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  @views solver.d = solver.fulld[m+1:m+n]
  return solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: MINRESSolver)

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  solver.fulld .= solver.in_solver.x
  @views solver.d = solver.fulld[m+1:m+n]
  return solver.d
end

"""
    Ared = ared(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, 
                rNorm :: AbstractFloat, rNormp :: AbstractFloat)

Calculate Ared = ‖F(xk)‖² - ‖F(xk+1)‖² in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function ared(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, 
              rNorm :: AbstractFloat, rNormp :: AbstractFloat)
  return rNorm^2 - rNormp^2
end


"""
    Pred = pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
                rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{true})

Calculate Pred = ‖F(xk)‖² - (‖J(xk)*d + F(xk)‖² + λ²‖d‖²) in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{true})
  mul!(solver.Ju, solver.Jx, solver.d)
  solver.Ju .= solver.Ju .+ solver.Fx
  normJu = norm(solver.Ju)
  return rNorm^2 - (normJu^2 + dNorm^2)
end

"""
    Pred = pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
                rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})

Calculate Pred = ‖F(xk)‖² - (‖J(xk)*d + F(xk)‖² + λ²‖d‖²) in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
  mul!(solver.Ju, solver.Jx, solver.d)
  solver.Ju .= solver.Ju .+ solver.Fx
  normJu = norm(solver.Ju)
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end

"""
    Pred = pred(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver}, 
                rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})

Calculate Pred = ‖F(xk)‖² - (‖J(xk)*d + F(xk)‖² + λ²‖d‖²) in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function pred(model :: AbstractNLSModel, solver :: Union{LDLSolver, MINRESSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
  m = model.nls_meta.nequ
  T = eltype(solver.x)
  @views fill!(solver.fulld[1:m], zero(T))
  mul!(solver.Ju, solver.A, solver.fulld)
#=   println("||J(x)*d||:")
  println(norm(solver.Ju[1:m]))
  println("||F(x)||:")
  println(norm(solver.Fx[1:m])) =#
  @views solver.Ju[1:m] .= solver.Ju[1:m] .+ solver.Fx[1:m]
  @views normJu = norm(solver.Ju[1:m])
#=   println("||d||^2:")
  println(dNorm^2)
  println("||λ||^2:")
  println(solver.λ^2)
  println("||J(x) *d + F(x)||^2:")
  println(normJu^2)
  println("||F(x)||^2:")
  println(rNorm^2) =#
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end

"""
    x, d, xp, solver = set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver, GPUSolver}, 
                                      TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)

Set x, d, xp, solver stored in the Levenberg-Marquardt solver.
"""
function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver, GPUSolver}, 
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  x, d, xp, solver = generic_solver.x, generic_solver.d, generic_solver.xp, generic_solver
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  x .= model.meta.x0
  return x, d, xp, solver
end

"""
    x, d, xp, solver = set_variables!(model :: AbstractNLSModel, generic_solver :: Union{MPSolver, MPGPUSolver},
                                      TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)

Set x, d, xp, solver stored in the Levenberg-Marquardt solver.
"""
function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{MPSolver, MPGPUSolver},
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  solver = generic_solver.F64Solver
  x, d, xp = solver.x, solver.d, solver.xp
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  copyto!(x, model.meta.x0)
  return x, d, xp, solver
end

"""
x, d, xp, solver = set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LDLSolver, MINRESSolver},
                                  TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)

Set x, d, xp, solver stored in the Levenberg-Marquardt solver.
"""
function set_variables!(model :: AbstractNLSModel, generic_solver :: Union{LDLSolver, MINRESSolver},
                        TR :: Bool, λ :: AbstractFloat, Δ :: AbstractFloat, λmin :: AbstractFloat)
  T = eltype(generic_solver.x)
  if TR
    error("Impossible to use trust region with LDL factorization")
  end
  if λ < 1e-10
    @printf("λ is too small, LDL factorization cannot be computed, setting λ to 1 instead")
    λ = one(T)
  end
  x, d, xp, solver = generic_solver.x, generic_solver.d, generic_solver.xp, generic_solver
  solver.TR, solver.λ, solver.Δ, solver.λmin = TR, λ, Δ, λmin
  x .= model.meta.x0
  return x, d, xp, solver
end

function minres_callback(model, solver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  T = eltype(solver.x)
  d2 = copy(solver.fulld)
  @views rNorm = norm(solver.Fx[1:m])
  @views fill!(d2[1:m], zero(T))
  @views dNorm = norm(d2[m+1:m+n])
  mul!(solver.Ju, solver.A, d2)
  @views solver.Ju[1:m] .= solver.Ju[1:m] .+ solver.Fx[1:m]
  @views normJu = norm(solver.Ju[1:m])
  if rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2) > 0
    return true
  end
  return false
end
