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
