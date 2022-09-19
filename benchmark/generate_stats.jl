using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2

"""
Function that scan the JLD2_files directory and generate stats tables and performance profiles
based on JLD2 archives, `solvers`, `costnames` and `costs`.
"""
function generate_stats(solvers :: AbstractVector,
                        costnames :: Vector{String}, 
                        costs :: AbstractVector, 
                        directory :: String = @__DIR__)
  
  # Creation of the stats variable where the stats are stored
  stats = Dict{Symbol, DataFrame}()

  for name in solvers
    stats[name] = DataFrame()
  end
  
  filenames = readdir(joinpath(directory, "..", "benchmark_files", "JLD2_files/"))
  
  # We open the JLD2 files and store the data in the stats dictionary
  for filename in filenames[2:end]
    file = jldopen(joinpath(directory, "..", "benchmark_files", "JLD2_files/", filename), "r")
    for name in solvers
      if String(name) in keys(file)
        solver_stats = file[String(name)]
        if size(stats[name]) == (0,0)
          stats[name] = similar(solver_stats, 0)
        end
        append!(stats[name], solver_stats)
      end
    end
    close(file)
  end

  # TODO : Add :dual_feas0, :residual0 and :inner_iter to the table
  # TODO : Add :neval_jprod_residual to the profile_solvers

  # We generate the stats table out of the stats dictionary
  for name in keys(stats)
    open(joinpath(directory, "..", "benchmark_files", "results", String(name) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".log"),"w") do io
      solver_df = stats[name]
      pretty_latex_stats(io, solver_df[!, [:name, :neval_residual, :inner_iter, :neval_jprod_residual, :status, :rNorm, :ArNorm0, :ArNorm, :elapsed_time]])
      pretty_stats(io, solver_df[!, [:name, :neval_residual, :inner_iter, :neval_jprod_residual, :status, :rNorm, :ArNorm0, :ArNorm, :elapsed_time]], tf=tf_markdown)
    end
  end

  # We generate the performance profile out of the stats dictionary
  profile_solvers(stats, costs, costnames)

  savefig(joinpath(directory, "..", "benchmark_files", "results", "performance_profile_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".pdf"))

end

# We choose the solvers
solvers = [:LM_facto]

# We define what a solved problem means
solved(stats) = map(x -> x in (:first_order, :small_residual), stats.status)

# We define what we want to compare in the performance profile
costnames = ["elapsed time", "num eval of residual", "num eval of jprod"]
costs = [stats -> .!solved(stats) .* Inf .+ stats.elapsed_time,
          stats -> .!solved(stats) .* Inf .+ stats.neval_residual,
          stats -> .!solved(stats) .* Inf .+ stats.neval_jprod_residual]

# We generate the files
generate_stats(solvers, costnames, costs)
