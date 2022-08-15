# This file is not up to date

using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Dates, DataFrames
ENV["GKSwstype"]="100"
using Plots
gr()

"""
Function that generates logging information and performance profile for a given set of `problems` 
with a given set of `solvers` and that compares differents `costs` with `costnames`
"""
function lm_benchmark(solvers :: Dict, 
                      problems :: DataFrame, 
                      costnames :: Vector{String}, 
                      costs :: AbstractVector, 
                      directory :: String = @__DIR__)

  # We generate the problem out of the problems dataframe
  problem_list = (BundleAdjustmentModel(problem[1],problem[2]) for problem in eachrow(problems))

  # We solve the problems with the given solvers
  stats = bmark_solvers(solvers, problem_list)

  # We generate the latex and markdown stats tables
  for solver in solvers
    open(joinpath(directory, String(solver.first) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".log"),"w") do io
      solver_df = stats[solver.first]
      pretty_latex_stats(io, solver_df[!, [:id, :name, :status, :objective, :dual_feas, :iter, :elapsed_time]])
      pretty_stats(io, solver_df[!, [:id, :name, :status, :objective, :dual_feas, :iter, :elapsed_time]], tf=tf_markdown)
    end
  end

  # We create and save the performance profile
  profile_solvers(stats, costs, costnames)

  savefig(joinpath(directory, "performance_profile_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".pdf"))

end

# We choose the problems
df = problems_df()
problems = df[( df.nnzj .≤ 1000000 ), :]

# We choose the solvers
solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model),
                :levenberg_marquardt_tr => model -> levenberg_marquardt_tr(model))

# We define what a solved problems means
solved(stats) = map(x -> x in (:first_order, :small_residual), stats.status)

# We choose what informations we want to compare in the performance profile
costnames = ["elapsed time", "num eval of residual"]
costs = [stats -> .!solved(stats) .* Inf .+ stats.elapsed_time,
          stats -> .!solved(stats) .* Inf .+ stats.neval_residual]

# We launch the benchmarks
lm_benchmark(solvers, problems, costnames, costs)
