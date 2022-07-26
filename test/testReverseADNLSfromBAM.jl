@testset "AD residual function test" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)

  Fx = residual(model, model.meta.x0)
  AD_Fx = residual(modelAD, modelAD.meta.x0)

  @test norm(Fx) ≈ norm(AD_Fx)
end

@testset "AD jacobian function test" begin
  model = BundleAdjustmentModel("problem-49-7776")
  modelAD = ADBundleAdjustmentModel(model)
  
  Fx = residual(model, model.meta.x0)
  S = typeof(model.meta.x0)
  meta_nls = nls_meta(model)
  rows = Vector{Int}(undef, meta_nls.nnzj)
  cols = Vector{Int}(undef, meta_nls.nnzj)
  vals = S(undef, meta_nls.nnzj)
  Jv = S(undef, meta_nls.nequ)
  Jtv = S(undef, meta_nls.nvar)
  jac_structure_residual!(model, rows, cols)
  jac_coord_residual!(model, model.meta.x0, vals)
  Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)

  AD_Fx = residual(modelAD, model.meta.x0)
  AD_Jv = S(undef, meta_nls.nequ)
  AD_Jtv = S(undef, meta_nls.nvar)
  AD_Jx = jac_op_residual!(modelAD, model.meta.x0, AD_Jv, AD_Jtv)

  @test norm(Jx' * Fx) ≈ norm(AD_Jx' * AD_Fx)
end