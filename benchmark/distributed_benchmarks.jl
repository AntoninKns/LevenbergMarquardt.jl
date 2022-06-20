using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2

function lm_distributed_benchmark(solvers :: Dict, 
                      partition_number :: Int, 
                      directory :: String = @__DIR__)

  problem_list = (BundleAdjustmentModel(problem[1],problem[2]) for problem in LevenbergMarquardt.partitions[partition_number])

  stats = bmark_solvers(solvers, problem_list)

  stats_JLD2 = joinpath(directory, "Partition_" * string(partition_number) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".jld2")

  jldopen(stats_JLD2, "w") do file
    for (name, solver) in solvers
      file[String(name)] = stats[name]
    end
  end

end

function main(args)
  solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model),
                :levenberg_marquardt_tr => model -> levenberg_marquardt_tr(model))

  partition_number = parse(Int64, args[1])

  lm_distributed_benchmark(solvers, partition_number)
end

main(ARGS)
