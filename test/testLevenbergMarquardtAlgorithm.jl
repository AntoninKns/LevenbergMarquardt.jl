
@testset "test residual norm reduction" begin
  model = BundleAdjustmentModel("problem-49-7776-pre.txt.bz2","ladybug")
  normFx0 = 1304.54011872448
  restol = normFx0/5
  objtol = (restol^2)/2
  stats = levenberg_marquardt(model, restol=restol)
  @test stats.objective ≤ objtol
end

@testset "test dual norm reduction" begin
  model = BundleAdjustmentModel("problem-49-7776-pre.txt.bz2","ladybug")
  x0 = model.meta.x0
  atol=sqrt(eps(eltype(x0)))
  rtol=eltype(x0)(eps(eltype(x0))^(1/3))
  normdual0 = 2.3961562909882326e7
  stats = levenberg_marquardt(model)
  @test stats.dual_feas ≤ atol + rtol*normdual0
end