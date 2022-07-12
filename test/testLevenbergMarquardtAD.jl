@testset "Levenberg-Marquardt Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  levenberg_marquardt_AD(modelAD, in_rtol = 1e-3, max_eval = 1, in_itmax = 1)
end

@testset "Levenberg-Marquardt Trust region Algorithm AD" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  levenberg_marquardt_tr_AD(modelAD, in_rtol = 1e-3, max_eval = 1, in_itmax = 1)
end