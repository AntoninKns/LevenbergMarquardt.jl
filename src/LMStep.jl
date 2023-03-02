"""

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, generic_solver :: Union{LMSolver, ADSolver, CGSolver})
  generic_solver.d .= generic_solver.in_solver.x
  # generic_solver.d .= generic_solver.in_solver.y
  return generic_solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: Union{MPSolver, GPUSolver, MPGPUSolver})

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, generic_solver :: MPSolver)
  # println("correcting step function")
  copyto!(generic_solver.F64Solver.d, generic_solver.F32Solver.in_solver.x)
  generic_solver.F64Solver.d .= generic_solver.F64Solver.d .+ generic_solver.F64Solver.in_solver.x
  return generic_solver.F64Solver.d
end

function step!(model :: AbstractNLSModel, generic_solver :: MPGPUSolver)
  # println("correcting step function")
  copyto!(generic_solver.F64Solver.d, generic_solver.F32Solver.in_solver.x)
  n = model.meta.nvar
  d = Vector{Float64}(undef, n)
  copyto!(d, generic_solver.F64Solver.in_solver.x)
  generic_solver.F64Solver.d .= generic_solver.F64Solver.d .+ d
  return generic_solver.F64Solver.d
end

function step!(model :: AbstractNLSModel, generic_solver :: GPUSolver)
  copyto!(generic_solver.d, generic_solver.in_solver.x)
  return generic_solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: LDLSolver)

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, generic_solver :: LDLSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  @views generic_solver.d = generic_solver.fulld[m+1:m+n]
  return generic_solver.d
end

"""
    solver.d = step!(model :: AbstractNLSModel, solver :: MINRESSolver)

Get the solution d from the subproblem.
"""
function step!(model :: AbstractNLSModel, generic_solver :: MINRESSolver)
  m = model.nls_meta.nequ
  n = model.meta.nvar
  generic_solver.fulld .= generic_solver.in_solver.x
  @views generic_solver.d = generic_solver.fulld[m+1:m+n]
  return generic_solver.d
end

function step!(model :: AbstractNLSModel, generic_solver :: SCHURSolver)
  return generic_solver.d
end

"""
    Ared = ared(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver}, 
                rNorm :: AbstractFloat, rNormp :: AbstractFloat)

Calculate Ared = ‖F(xk)‖² - ‖F(xk+1)‖² in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function ared(solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, LDLSolver, MINRESSolver, CGSolver, SCHURSolver}, 
              rNorm :: AbstractFloat, rNormp :: AbstractFloat)
  return rNorm^2 - rNormp^2
end


"""
    Pred = pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver}, 
                rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{true})

Calculate Pred = ‖F(xk)‖² - (‖J(xk)*d + F(xk)‖² + ‖d‖²) in order to obtain ρ = Ared / Pred which determines the quality of the step.
"""
function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, CGSolver}, 
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
function pred(model :: AbstractNLSModel, solver :: Union{LMSolver, MPSolver, ADSolver, GPUSolver, MPGPUSolver, CGSolver}, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
  mul!(solver.Ju, solver.Jx, solver.d)
  solver.Ju .= solver.Ju .+ solver.Fx
  normJu = norm(solver.Ju)
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end

function pred(model :: AbstractNLSModel, solver :: SCHURSolver, 
              rNorm :: AbstractFloat, dNorm :: AbstractFloat, :: Val{false})
mul!(solver.Ju, solver.Jx, solver.d)
solver.Ju .= solver.Ju .+ solver.Fx
normJu = norm(solver.Ju)
return rNorm^2 - normJu^2
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
  @views solver.Ju[1:m] .= solver.Ju[1:m] .+ solver.Fx[1:m]
  @views normJu = norm(solver.Ju[1:m])
  return rNorm^2 - (normJu^2 + solver.λ^2*dNorm^2)
end
