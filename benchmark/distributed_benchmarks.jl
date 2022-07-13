using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2

"""
Function that solves problems based on their partition number and the partition list and saves the stats in a JLD2 file.
"""
function lm_distributed_benchmark(solvers :: Dict, 
                      partition_number :: Int, 
                      directory :: String = @__DIR__)

  problem_list = (BundleAdjustmentModel(problem) for problem in LevenbergMarquardt.partitions[partition_number])

  stats = bmark_solvers(solvers, problem_list)

  stats_JLD2 = joinpath(directory, "JLD2_files", "Partition_" * string(partition_number) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".jld2")

  jldopen(stats_JLD2, "w") do file
    for (name, solver) in solvers
      file[String(name)] = stats[name]
    end
  end

end

# Get the solver and partition number and launch the distributed benchmark
function main(args)
  solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model, in_rtol=1e-3),
                :levenberg_marquardt_AD => model -> levenberg_marquardt_AD_BAM(model, in_rtol=1e-3))

  partition_number = parse(Int64, args[1])

  lm_distributed_benchmark(solvers, partition_number)
end

main(ARGS)
