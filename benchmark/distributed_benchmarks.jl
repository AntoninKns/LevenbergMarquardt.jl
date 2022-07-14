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

function bmark_solvers_lm(solvers::Dict{Symbol, <:Any}, args...; kwargs...)
  stats = Dict{Symbol, DataFrame}()
  for (name, solver) in solvers
    @debug "running" name solver
    stats[name] = solve_problems_lm(solver, args...; kwargs...)
  end
  return stats
end

function solve_problems_lm(solver, problems; 
                            solver_logger :: AbstractLogger = NullLogger(),
                            reset_problem :: Bool = true,
                            skipif :: Function = x -> false,
                            prune :: Bool = true)

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

  col_idx = indexin(colstats, names)

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
        s = with_logger(solver_logger) do
          solver(problem; kwargs...)
        end
        if first_problem
          for (k, v) in s.solver_specific
            if !(typeof(v) <: AbstractVector)
              insertcols!(stats, ncol(stats) + 1, k => Vector{Union{typeof(v), Missing}}())
              push!(specific, k)
            end
          end

          @info log_header(colstats, types[col_idx], hdr_override = info_hdr_override)

          first_problem = false
        end

        @printf("| %17s | %6s | %6s | %14s | %8s | %8s | %8s | %8s |\n", "Name", "nvar", "nequ", "status", "rNorm0", "rNorm", "ArNorm0", "ArNorm")

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
            [getfield(s.counters.counters, f) for f in f_counters]
            [getfield(s.counters, f) for f in fnls_counters]
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
    (skipthis && prune) || @printf("| %17s | %6d | %6d | %14s | %1.2e | %1.2e | %1.2e | %1.2e |\n", 
                                    problem.meta.name, problem.meta.nvar, nequ, String(s.status), 
                                    s.rNorm0, s.rNorm, s.ArNorm0, s.ArNorm)
  end
  return stats
end

# Get the solver and partition number and launch the distributed benchmark
function main(args)
  solvers = Dict(:levenberg_marquardt => model -> levenberg_marquardt(model, in_rtol=1e-3),
                :levenberg_marquardt_AD => model -> levenberg_marquardt_AD_BAM(model, in_rtol=1e-3))

  partition_number = parse(Int64, args[1])

  lm_distributed_benchmark(solvers, partition_number)
end

main(ARGS)
