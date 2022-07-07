using SolverBenchmark, LevenbergMarquardt, BundleAdjustmentModels, Plots, Dates, DataFrames, JLD2

# We choose the solvers
old_solvers = [:levenberg_marquardt, :levenberg_marquardt_tr]
new_solvers = [:levenberg_marquardt_1e3, :levenberg_marquardt_tr_1e3]

# We choose the directory
directory = @__DIR__

filenames = readdir(joinpath(directory, "JLD2_files/"))
  
# We open the JLD2 files and store the data in the stats dictionary
for filename in filenames
  file = jldopen(joinpath(directory, "JLD2_files/", filename), "r")
  filename2 = "2_"*filename
  file2 = jldopen(joinpath(directory, "JLD2_files_2/", filename2), "w")
  for i in 1:length(old_solvers)
    solver_stats = file[String(old_solvers[i])]
    file2[String(new_solvers[i])] = solver_stats
  end
  close(file)
  close(file2)
end
