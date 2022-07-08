@testset "LevenbergMarquardtAlgorithm allocations" begin
  model = SimpleNLSModel()
  solver = LMSolver(model)
  levenberg_marquardt!(solver, model, verbose = false)
  levenberg_marquardt_alloc(solver, model) =
    @allocated levenberg_marquardt!(solver, model, verbose = false)
  @test levenberg_marquardt_alloc(solver, model) == 0
end