using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2

function generate_stats(solvers :: Dict,
                        costnames :: Vector{String}, 
                        costs :: AbstractVector, 
                        directory :: String = @__DIR__)
  
  stats = Dict{Symbol, DataFrame}()

  for (name, solver) in solvers
    stats[name] = DataFrame()
  end
  
  filenames = readdir(joinpath(directory, "JLD2_files/"))
  
  for filename in filenames
    file = jldopen(joinpath(directory, "JLD2_files/", filename), "r")
    for (name, solver) in solvers
      println(String(name))
      solver_stats = file[String(name)]
      if size(stats[name]) == (0,0)
        stats[name] = similar(solver_stats, 0)
      end
      append!(stats[name], solver_stats)
    end
    close(file)
  end

  for solver in solvers
    open(joinpath(directory, "results", String(solver.first) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".log"),"w") do io
      solver_df = stats[solver.first]
      pretty_latex_stats(io, solver_df[!, [:name, :nequ, :nvar, :neval_residual, :status, :objective, :dual_feas, :iter, :elapsed_time]])
      pretty_stats(io, solver_df[!, [:name, :nequ, :nvar, :neval_residual, :status, :objective, :dual_feas, :iter, :elapsed_time]], tf=tf_markdown)
    end
  end

  profile_solvers(stats, costs, costnames)

  savefig(joinpath(directory, "results", "performance_profile_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".pdf"))

end

solved(stats) = map(x -> x in (:first_order, :small_residual), stats.status)

solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model),
                :levenberg_marquardt_tr => model -> levenberg_marquardt_tr(model))

costnames = ["elapsed time", "num eval of residual"]

costs = [stats -> .!solved(stats) .* Inf .+ stats.elapsed_time,
          stats -> .!solved(stats) .* Inf .+ stats.neval_residual]

generate_stats(solvers, costnames, costs)
