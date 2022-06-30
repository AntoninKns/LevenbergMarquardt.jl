export levenberg_marquardt_tr

"""
Algorithm of Levenberg Marquardt based on "
"""

function levenberg_marquardt_tr(nls :: AbstractNLSModel;
                              x :: AbstractVector = copy(nls.meta.x0),
                              η₁ :: AbstractFloat = eltype(x)(1e-4), 
                              η₂ :: AbstractFloat = eltype(x)(0.99),
                              σ₁ :: AbstractFloat = eltype(x)(3.0), 
                              σ₂ :: AbstractFloat = eltype(x)(1/3),
                              max_eval :: Int = 10_000,
                              Δ :: AbstractFloat = eltype(x)(10.),
                              restol=eltype(x)(eps(eltype(x))^(1/3)),
                              atol=sqrt(eps(eltype(x))), 
                              rtol=eltype(x)(eps(eltype(x))^(1/3)))


  # We set up the initial value of the residual and Jacobien based on the starting point
  Fx = residual(nls, x)
  S = typeof(nls.meta.x0)
  meta_nls = nls_meta(nls)
  rows = Vector{Int}(undef, meta_nls.nnzj)
  cols = Vector{Int}(undef, meta_nls.nnzj)
  vals = S(undef, meta_nls.nnzj)
  Jv = S(undef, meta_nls.nequ)
  Jtv = S(undef, meta_nls.nvar)

  jac_structure_residual!(nls, rows, cols)
  jac_coord_residual!(nls, x, vals)
  Jx = jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)
  Fxp = similar(Fx)

  normFx = normFx0 = norm(Fx)
  normdual = normdual0 = norm(Jx'*Fx)

  # We set up the initial parameters 
  iter = 0
  T = eltype(x)
  start_time = time()
  solver_specific = Dict(:inner_iter => 0, 
  :dual_feas0 => normdual0,
  :objective0 => normFx0^2/2)

  optimal = false
  small_residual = false
  tired = false

  levenberg_marquardt_tr_log_header(nls)

  while !(optimal || small_residual || tired )

    # We solve the subproblem
    d, inner_stats = lsmr(Jx, -Fx, radius = T(Δ), axtol=1e-3, btol=1e-3, atol=1e-3, rtol=1e-3, etol=1e-3)

    xp      = x + d
    Fxp = residual!(nls, xp, Fxp)
    normFxp = norm(Fxp)

    # We test the quality of the state
    Pred = (normFx^2 - norm(Jx * d + Fx)^2)/2
    Ared = (normFx^2 - normFxp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step we update the step and/or the parameters
    if ρ < η₁
      Δ = Δ * σ₁
    else
      x  .= xp
      jac_coord_residual!(nls, x, vals)
      Jx = jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)
      Fx = Fxp
      normFx = normFxp
      Jtr = Jx'*Fx
      normdual = norm(Jtr)
      if ρ > η₂
        Δ = Δ * σ₂
      end
    end

    # We update the logging informations
    inner_status = change_stats(inner_stats.status)
    iter += 1
    solver_specific[:inner_iter] += inner_stats.niter
    levenberg_marquardt_tr_log_row(iter, (normFx^2)/2, normdual, norm(d), Δ, Ared, Pred, ρ, inner_status, inner_stats.niter, neval_jprod_residual(nls))

    # We update the stopping conditions
    optimal = normdual < atol + rtol*normdual0
    tired = neval_residual(nls) > max_eval
    small_residual = normFx < restol

  end

  # We update the last parameters
  el_time = time()-start_time

  # we update the status of the solver
  if optimal 
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif tired
    status = :max_iter
  else
    status = :unknown
  end

  # We return all the logs in a dedicated structure
  return GenericExecutionStats(status, nls, solution = x,
                                objective = obj(nls, x),
                                dual_feas = normdual,
                                iter = iter, 
                                elapsed_time = el_time,
                                solver_specific = solver_specific)
end
