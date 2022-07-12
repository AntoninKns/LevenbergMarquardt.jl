@testset "Levenberg-Marquardt Algorithm allocations" begin
  model = BundleAdjustmentModel("problem-49-7776")
  solver = LMSolver(model)
  levenberg_marquardt!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  levenberg_marquardt_alloc(solver, model) =
    @allocated levenberg_marquardt!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  @test levenberg_marquardt_alloc(solver, model) <= 10000
end

@testset "Levenberg-Marquardt Trust Region allocations" begin
  model = BundleAdjustmentModel("problem-49-7776")
  solver = LMSolver(model)
  levenberg_marquardt_tr!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  levenberg_marquardt_tr_alloc(solver, model) =
    @allocated levenberg_marquardt_tr!(solver, model, verbose = false, max_eval = 1, in_itmax = 1)
  @test levenberg_marquardt_tr_alloc(solver, model) <= 10000
end