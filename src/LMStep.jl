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
