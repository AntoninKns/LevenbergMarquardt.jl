export levenberg_marquardt_tr

"""
Algorithm of Levenberg Marquardt based on "
"""

function levenberg_marquardt_tr(nls :: AbstractNLSModel;
                              x :: AbstractVector = copy(nls.meta.x0),
                              η₁ :: AbstractFloat = eltype(x)(1e-4), 
                              η₂ :: AbstractFloat = eltype(x)(0.99),
                              σ₁ :: AbstractFloat = eltype(x)(10.0), 
                              σ₂ :: AbstractFloat = eltype(x)(0.66),
                              max_eval :: Int = 10_000,
                              λmin :: AbstractFloat = eltype(x)(1e-1),
                              Δ :: AbstractFloat = eltype(x)(10.),
                              restol=eltype(x)(eps(eltype(x))^(1/3)),
                              atol=sqrt(eps(eltype(x))), 
                              rtol=eltype(x)(eps(eltype(x))^(1/3)))


  # We set up the initial value of the residual and Jacobien based on the starting point
  Fx = residual(nls, x)
  Jx = jac_op_residual(nls, x)
  Fxp = similar(Fx)

  normFx = norm(Fx)
  normdual = normdual0 = norm(Jx'*Fx)

  # We set up the initial parameters 
  iter = 0
  λ = 0.
  T = eltype(x)
  start_time = time()

  optimal = false
  small_residual = false
  tired = false

  # This it the logging bar to have information about the state of the algorithm
  @info log_header([:outer_iter, :obj, :dual, :nd, :λ, :Ared, :Pred, :ρ, :inner_status, :Δ],
  [Int, T, T, T, T, T, T, T, String, T],
  hdr_override=Dict(:obj => "‖F(x)‖²/2", :dual => "‖J'F‖", :nd => "‖d‖"))

  while !(optimal || small_residual || tired )

    # We solve the subproblem
    d, inner_stats = lsmr(Jx, -Fx, λ = T(λ), radius = T(Δ))

    xp      = x + d
    Fxp = residual!(nls, xp, Fxp)
    normFxp = norm(Fxp)

    # We test the quality of the state
    Pred = (normFx^2 - norm(Jx * d + Fx)^2 -λ^2*norm(d)^2)/2
    Ared = (normFx^2 - normFxp^2)/2
    ρ = Ared/Pred

    # Depending on the quality of the step we update the step and/or the parameters
    if ρ < η₁
      if λ == 0
        λ = λmin
      else
        λ = σ₁ * λ
      end
      Δ = Δ/3
    else
      x  .= xp
      jac_coord!(nls, x, nls.vals)
      Jx = jac_op_residual!(nls, nls.rows, nls.cols, nls.vals, nls.Jv, nls.Jtv)
      Fx = Fxp
      normFx = normFxp
      Jtr = Jx'*Fx
      normdual = norm(Jtr)
      if ρ > η₂
        λ = σ₂ * λ
        Δ = Δ*3
      end
      if λ < λmin
        λ = 0
      end
    end

    # We update the logging informations
    inner_status = inner_stats.status
    iter += 1
    @info log_row(Any[iter, (normFx^2)/2, normdual, norm(d), λ, Ared, Pred, ρ, inner_status, Δ])

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
                                elapsed_time = el_time)
end
