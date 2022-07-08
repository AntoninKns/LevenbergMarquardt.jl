@testset "simple-model residual" begin
  model = SimpleNLSModel()
  x = model.meta.x0
  Fx = similar(x, model.nls_meta.nequ)
  residual!(model, x, Fx)
  residual_alloc(model, x, Fx) =
    @allocated residual!(model, x, Fx)
  @test residual_alloc(model, x, Fx) == 0
end

@testset "simple-model jac_structure_residual" begin
  model = SimpleNLSModel()
  x = model.meta.x0
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  jac_structure_residual_alloc(model, rows, cols) =
    @allocated jac_structure_residual!(model, rows, cols)
  @test jac_structure_residual_alloc(model, rows, cols) == 0
end

@testset "simple-model jac_coord_residual" begin
  model = SimpleNLSModel()
  x = model.meta.x0
  vals = similar(x, model.nls_meta.nnzj)
  jac_coord_residual!(model, x, vals)
  jac_coord_residual_alloc(model, x, vals) =
    @allocated jac_coord_residual!(model, x, vals)
  @test jac_coord_residual_alloc(model, x, vals) == 0
end
