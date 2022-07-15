using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2, Printf, NLPModels

"""
Function that solves problems based on their partition number and the partition list and saves the stats in a JLD2 file.
"""
function lm_distributed_benchmark(solvers :: Dict, 
                      partition_number :: Int, 
                      directory :: String = @__DIR__)

  problem_list = (BundleAdjustmentModel(problem) for problem in LevenbergMarquardt.partitions[partition_number])

  stats = bmark_solvers_lm(solvers, problem_list)

  stats_JLD2 = joinpath(directory, "JLD2_files", "Partition_" * string(partition_number) * "_stats_" * Dates.format(now(), DateFormat("yyyymmddHMS")) * ".jld2")

  jldopen(stats_JLD2, "w") do file
    for (name, solver) in solvers
      file[String(name)] = stats[name]
    end
  end

end

function bmark_solvers_lm(solvers::Dict{Symbol, <:Any}, args...; kwargs...)
  stats = Dict{Symbol, DataFrame}()
  for (name, solver) in solvers
    @debug "running" name solver
    stats[name] = solve_problems_lm(solver, args...; kwargs...)
  end
  return stats
end

function solve_problems_lm(solver, problems; 
                            reset_problem :: Bool = true,
                            skipif :: Function = x -> false,
                            prune :: Bool = true,
                            kwargs...)

  f_counters = collect(fieldnames(Counters))
  fnls_counters = collect(fieldnames(NLSCounters))[2:end] # Excludes :counters
  ncounters = length(f_counters) + length(fnls_counters)

  types = [
    String
    Int
    Int
    Symbol
    Float64
    Float64
    Float64
    Float64
    Int
    Int
    Float64
    fill(Int, ncounters)
  ]

  names = [
    :name
    :nvar
    :nequ
    :status
    :rNorm
    :rNorm0
    :ArNorm
    :ArNorm0
    :iter
    :inner_iter
    :elapsed_time
    f_counters
    fnls_counters
  ]

  stats = DataFrame(names .=> [T[] for T in types])

  specific = Symbol[]

  first_problem = true
  for (id, problem) in enumerate(problems)
    if reset_problem
      reset!(problem)
    end
    nequ = problem isa AbstractNLSModel ? problem.nls_meta.nequ : 0
    problem_info = [problem.meta.name; problem.meta.nvar; nequ]
    skipthis = skipif(problem)
    if skipthis
      prune || push!(
        stats,
        [
          problem_info
          :exception
          Inf
          Inf
          Inf
          Inf
          0
          0
          Inf
          fill(0, ncounters)
        ],
      )
      finalize(problem)
    else
      try
        s = solver(problem; kwargs...)

        @printf("| %17s | %6s | %6s |\n", "Name", "nvar", "nequ")

        push!(
          stats,
          [
            problem_info
            s.status
            s.rNorm0
            s.rNorm
            s.ArNorm0
            s.ArNorm
            s.iter
            s.inner_iter
            s.elapsed_time
            [getfield(s.model.counters.counters, f) for f in f_counters]
            [getfield(s.model.counters, f) for f in fnls_counters]
          ],
        )
      catch e
        @error "caught exception" e
        push!(
          stats,
          [
            problem_info
            :exception
            Inf
            Inf
            Inf
            Inf
            0
            0
            Inf
            fill(0, ncounters)
          ],
        )
      finally
        finalize(problem)
      end
    end
    (skipthis && prune) || @printf("| %17s | %6d | %6d |\n", problem.meta.name, problem.meta.nvar, nequ)
  end
  return stats
end

# Get the solver and partition number and launch the distributed benchmark
function main(args)
  solvers = Dict(:levenberg_marquardt_AD => model -> levenberg_marquardt_AD_BAM(model, in_rtol=1e-3),
                :levenberg_marquardt => model -> levenberg_marquardt(model, in_rtol=1e-3)
                )

  partition_number = parse(Int64, args[1])

  lm_distributed_benchmark(solvers, partition_number)
end

main(ARGS)
