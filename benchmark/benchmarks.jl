using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates

function lm_benchmark(solvers :: Dict, 
                      problems :: DataFrame, 
                      costnames :: Vector{String}, 
                      costs :: AbstractVector, 
                      directory :: String = @__DIR__)

  problem_list = (BundleAdjustmentModel(problem[1],problem[2]) for problem in eachrow(problems))

  stats = bmark_solvers(solvers, problem_list)

  for solver in solvers
    open(joinpath(directory, String(solver.first) * "_stats_" *string(Dates.now())* ".log","w")) do io
      solver_df = stats[solver.first]
      pretty_latex_stats(io, solver_df[!, [:id, :name, :status, :objective, :dual_feas, :iter, :elapsed_time]])
      pretty_stats(io, solver_df[!, [:id, :name, :status, :objective, :dual_feas, :iter, :elapsed_time]], tf=markdown)
    end
  end

  profile_solvers(stats, costs, costnames)

  savefig(joinpath(directory, "performance_profile_" * string(Dates.now()) * ".pdf"))

end

solved(stats) = map(x -> x in (:first_order, :small_residual), stats.status)

solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model),
                :levenberg_marquardt_tr => model -> levenberg_marquardt_tr(model))

df = problems_df()
problems = df[( df.nnzj .≤ 70000 )]

costnames = ["elapsed time", "num eval of residual"]

costs = [stats -> .!solved(stats) .* Inf .+ stats.elapsed_time,
          stats -> .!solved(stats) .* Inf .+ stats.neval_residual]

lm_benchmark(solvers, problems, costnames, costs)
