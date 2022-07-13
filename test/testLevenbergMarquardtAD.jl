@testset "Levenberg-Marquardt Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  solver = LMSolverAD(modelAD)
  levenberg_marquardt_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  @test solver.stats.rNorm ≤ solver.stats.rNorm0
end

@testset "Levenberg-Marquardt Trust region Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  solver = LMSolverAD(modelAD)
  levenberg_marquardt_tr_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  @test solver.stats.rNorm ≤ solver.stats.rNorm0
end