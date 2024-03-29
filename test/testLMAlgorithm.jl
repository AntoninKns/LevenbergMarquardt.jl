@testset "test levenberg_marquardt" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model)
  @test stats.rNorm ≤ 1e-4
end

@testset "test levenberg_marquardt trust region" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, TR = true)
  @test stats.rNorm ≤ 1e-4
end

@testset "test residual norm reduction" begin
  model = SimpleNLSModel()
  normFx0 = norm(residual(model, model.meta.x0))
  restol = normFx0/100
  stats = levenberg_marquardt(model, restol=restol)
  @test stats.rNorm ≤ restol
end

@testset "test dual norm reduction" begin
  model = SimpleNLSModel()
  x0 = model.meta.x0
  atol=1e-2
  rtol=1e-2
  jac_structure_residual!(model, model.rows, model.cols)
  jac_coord_residual!(model, x0, model.vals)
  Jv = similar(x0, model.nls_meta.nequ)
  Jtv = similar(x0, model.meta.nvar)
  Jx = jac_op_residual!(model, model.rows, model.cols, model.vals, Jv, Jtv)
  Fx = residual(model, x0)
  ArNorm0 = norm(Jx * Fx)
  stats = levenberg_marquardt(model, atol=atol, rtol=rtol)
  @test stats.ArNorm ≤ atol + rtol*ArNorm0
end

@testset "test GPU algorithm" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, which = :GPU)
  @test stats.status == :first_order || stats.status == :small_residual
end

@testset "test LDL algorithm" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, which = :LDL)
  @test stats.status == :first_order || stats.status == :small_residual
end

@testset "test MINRES algorithm" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, which = :MINRES)
  @test stats.status == :first_order || stats.status == :small_residual
end

@testset "test MPGPU algorithm" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, which = :MPGPU)
  @test stats.status == :first_order || stats.status == :small_residual
end

@testset "test MP algorithm" begin
  model = SimpleNLSModel()
  stats = levenberg_marquardt(model, which = :MP)
  @test stats.status == :first_order || stats.status == :small_residual
end
