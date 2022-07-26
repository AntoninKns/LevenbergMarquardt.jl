@testset "Levenberg-Marquardt Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  solver = LMSolverAD(modelAD)
  levenberg_marquardt_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  function levenberg_marquardt_AD_alloc(solver, modelAD)
    @allocated levenberg_marquardt_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  end
  @test levenberg_marquardt_AD_alloc(solver, modelAD) <= 10000
end

@testset "Levenberg-Marquardt Trust region Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  solver = LMSolverAD(modelAD)
  levenberg_marquardt_tr_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  levenberg_marquardt_tr_AD_alloc(solver, modelAD) =
    @allocated levenberg_marquardt_tr_AD!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  @test levenberg_marquardt_tr_AD_alloc(solver, modelAD) <= 10000
end