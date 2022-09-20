@testset "Levenberg-Marquardt Algorithm allocations" begin
  model = BundleAdjustmentModel("problem-49-7776")
  solver = LMSolver(model)
  levenberg_marquardt!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  levenberg_marquardt_alloc(solver, model) =
    @allocated levenberg_marquardt!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  @test levenberg_marquardt_alloc(solver, model) >= 1000
end

@testset "Levenberg-Marquardt Trust Region allocations" begin
  model = BundleAdjustmentModel("problem-49-7776")
  solver = LMSolver(model)
  levenberg_marquardt!(solver, model, TR = true, verbose = false, max_eval = 1, in_itmax = 1)
  levenberg_marquardt_tr_alloc(solver, model) =
    @allocated levenberg_marquardt!(solver, model, TR = true, verbose = false, max_eval = 1, in_itmax = 1)
  @test levenberg_marquardt_tr_alloc(solver, model) >= 1000
end

@testset "Levenberg-Marquardt Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  solver = LMSolverAD(modelAD)
  levenberg_marquardt!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  function levenberg_marquardt_AD_alloc(solver, modelAD)
    @allocated levenberg_marquardt!(solver, modelAD, verbose = false, max_eval = 1, in_itmax = 1)
  end
  @test levenberg_marquardt_AD_alloc(solver, modelAD) >= 1000000
end
