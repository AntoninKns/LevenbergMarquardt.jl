# How to use scripts to launch benchmarks on forntal22

## Setup environment

You can either fully run your scripts and benchmark on frontal22 or launch the tests on frontal22 and retrieve tne results on your local machine.

In order to launch the test you need to have acces to frontal22 and in your personal directory launch the command 

```
git clone https://github.com/AntoninKns/LevenbergMarquardt.jl.git
```

Once it is done go to the LevenbergMarquardt.jl directory

## Launch the benchmarks

The benchmarks are launched from the LevenbergMarquardt.jl directory using the ``run_benchmarks.sh`` script. It will launch the 21 partition scripts that will then launch the ``distributed_benchmarks.jl`` file.

You can change the partitions by modifying the ``Partitions.jl`` file and then adapting the different scripts.

## Retrieve the JLD2 files

Once the benchmarks have generated the JLD2 files you can either run the ``retrieve_files_windows.sh`` on your local windows machine, or run the equivalent scp command on a linux machine.

## Generate the stats

Once you have got the JLD2 files on the machine where you want to generate the stats tables and performance profiles you can type ``include("benchmark/generate_stats.jl")`` from a julia REPL in the LevenbergMarquardt.jl directory.

## Clean environment

Once anything is over and you have copied the output and error files you wanted, you can run the ``clean_benchmarks.sh`` files. This will delete all the error and output files but not the JLD2 files nor the stats tables and performance profiles.
