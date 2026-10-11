# Compare solver variants on a set of CUTEst problems with SolverBenchmark.
#
# Usage:
#   julia benchmark/solver_benchmarks.jl [outdir]
#
#   outdir  directory where results are written (default: benchmark_results)
#
# Set `NLPMODELSIPOPT_BMARK_MAX_PROBLEMS` to only solve the first problems of the set.
using Pkg

const bmark_dir = @__DIR__
const pkg_dir = normpath(joinpath(bmark_dir, ".."))

outdir = abspath(length(ARGS) ≥ 1 ? ARGS[1] : "benchmark_results")

const tmp_dir = mktempdir()
cp(joinpath(bmark_dir, "Project.toml"), joinpath(tmp_dir, "Project.toml"))
Pkg.activate(tmp_dir)
Pkg.develop(PackageSpec(path = pkg_dir))
Pkg.instantiate()

using CUTEst
using DataFrames
using JLD2
using NLPModelsIpopt
using Plots
using SolverBenchmark

problem_names = sort(select_sif_problems(max_var = 10, max_con = 10))
if haskey(ENV, "NLPMODELSIPOPT_BMARK_MAX_PROBLEMS")
  max_problems = parse(Int, ENV["NLPMODELSIPOPT_BMARK_MAX_PROBLEMS"])
  problem_names = first(problem_names, max_problems)
end

# re-iterating the generator creates fresh models for each solver
problems = (CUTEstModel(name) for name ∈ problem_names)

solvers = Dict{Symbol, Function}(
  :ipopt => nlp -> ipopt(nlp, print_level = 0),
  :ipopt_lbfgs => nlp -> ipopt(nlp, print_level = 0, hessian_approximation = "limited-memory"),
)

stats = bmark_solvers(solvers, problems)
mkpath(outdir)
save_stats(stats, joinpath(outdir, "solver_stats.jld2"), force = true)

# unsolved problems get an infinite cost
solved(df) = df.status .== :first_order
costs = [
  df -> ifelse.(solved(df), df.elapsed_time, Inf),
  df -> ifelse.(solved(df), df.neval_obj .+ df.neval_cons, Inf),
  df -> ifelse.(solved(df), df.iter, Inf),
]
profile_solvers(stats, costs, ["time", "#f + #c", "iterations"])
savefig(joinpath(outdir, "solver_profiles.svg"))
savefig(joinpath(outdir, "solver_profiles.png"))

open(joinpath(outdir, "summary.md"), "w") do io
  println(
    io,
    "Benchmarked on $(length(problem_names)) CUTEst problems with at most 10 variables and 10 constraints.\n",
  )
  println(io, "| solver | solved | total time (s) | total iterations |")
  println(io, "|--------|-------:|---------------:|-----------------:|")
  for (solver, df) ∈ sort(collect(stats), by = first)
    println(
      io,
      "| $solver | $(count(solved(df))) | $(round(sum(df.elapsed_time), digits = 2)) | $(sum(df.iter)) |",
    )
  end
end

println("Benchmark results written to $outdir")
