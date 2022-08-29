using LevenbergMarquardt, BundleAdjustmentModels, Dates, DataFrames, JLD2

function greedy_partition(elements :: DataFrame, n_partitions :: Integer, criteria :: Symbol)

  partition_list = [Vector{Tuple{String,Number}}(undef,0) for _ in 1:n_partitions]

  sort!(elements, [order(criteria, rev=true)])

  n = size(elements, 1)

  for i in 1:n
    element = (elements[i,1], elements[i,criteria])
    sum_list = [sum([x[2] for x in partition_list[i]]) for i in 1:n_partitions]
    min_idx = findmin(sum_list)[2]
    push!(partition_list[min_idx], element)
  end

  partition = [[x[1] for x in partition_list[i]] for i in 1:n_partitions]

  return partition

end

function generate_partitions(solvers :: AbstractVector,
                            n_partitions :: Integer,
                            criteria :: Symbol = :elapsed_time,
                            solver :: Symbol = solvers[1],
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

  partition = greedy_partition(stats[solver], n_partitions, criteria)

  return partition

end

# We choose the solvers
solvers = [:LM, :LM_TR]
n_partitions = 5

generate_partitions(solvers, n_partitions)
