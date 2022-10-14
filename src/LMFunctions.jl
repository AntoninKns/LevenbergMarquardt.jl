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
