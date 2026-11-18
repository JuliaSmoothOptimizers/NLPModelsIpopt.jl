# Solve a set of CUTEst problems with each solver variant.
#
# JSOBenchmarks.run_solver_benchmarks runs this script on the current commit and on the
# reference branch, and compares the results of each solver.
#
# Set `NLPMODELSIPOPT_BMARK_MAX_PROBLEMS` to only solve the first problems of the set.
using CUTEst
using JSOBenchmarks
using NLPModelsIpopt
using SolverBenchmark

JSOBenchmarks.solver_benchmark_profile_values() =
  [(:elapsed_time, "time"), (:iter, "iterations"), (:neval_obj, "#f"), (:neval_cons, "#c")]
JSOBenchmarks.solver_benchmark_table_values() =
  [(:name, "name"), (:status, "status"), (:objective, "f(x)"), (:iter, "iter"), (:elapsed_time, "time")]

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

bmark_solvers(solvers, problems)
