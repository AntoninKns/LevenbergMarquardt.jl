@testset "test levenberg_marquardt_tr" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt_tr(model)
  @test stats.objective ≤ 1e-5
end
