
@testset "test residual norm reduction" begin
  model = BundleAdjustmentModel("problem-1778-993923-pre.txt.bz2","venice")
  normFx0 = 22644.967624342433
  restol = normFx0
  objtol = (restol^2)/2
  stats = levenberg_marquardt(model, restol=restol)
  @test stats.objective ≤ objtol
end

@testset "test dual norm reduction" begin
  model = BundleAdjustmentModel("problem-1778-993923-pre.txt.bz2","venice")
  x0 = model.meta.x0
  atol=sqrt(eps(eltype(x0)))
  rtol=eltype(x0)(eps(eltype(x0))^(1/3))
  normdual0 = 3.190087189251177e15
  stats = levenberg_marquardt(model)
  @test stats.dual_feas ≤ atol + rtol*normdual0
end
