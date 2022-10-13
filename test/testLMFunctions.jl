@testset "test pred" begin
  model = BundleAdjustmentModel("problem-49-7776")
  m = model.nls_meta.nequ
  n = model.meta.nvar
  x = model.meta.x0
  T = eltype(x)
  solverLM = LMSolver(model)
  solverMINRES = MINRESSolver(model)
  residualLM!(model, x, solverLM)
  set_jac_residual!(model, x, solverLM)
  residualLM!(model, x, solverMINRES)
  set_jac_residual!(model, x, solverMINRES)
  solverLM.d = ones(n)
  solverMINRES.fulld = ones(m+n)
  solverMINRES.d = ones(n)
  solverLM.λ = one(T)
  solverMINRES.λ = one(T)
  dNormLM = norm(solverLM.d)
  dNormMINRES = norm(solverMINRES.d)
  rNormLM = rNorm!(solverLM)
  rNormMINRES = rNorm!(solverMINRES)
  predLM = pred(model, solverLM, rNormLM, dNormLM)
  predMINRES = pred(model, solverMINRES, rNormMINRES, dNormMINRES)
  @test predLM ≈ predMINRES
end