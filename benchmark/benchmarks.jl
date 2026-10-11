using BenchmarkTools

using CUTEst
using NLPModelsIpopt

const SUITE = BenchmarkGroup()

# Set the environment variable `NLPMODELSIPOPT_BMARK_PROBLEMS` to a comma-separated
# list of CUTEst problem names to benchmark a different set, e.g., "ROSENBR,WOODS".
const DEFAULT_PROBLEMS = ["ROSENBR", "WOODS", "PENALTY1", "HS6", "HS21", "HS35", "HS71"]
const PROBLEMS = if haskey(ENV, "NLPMODELSIPOPT_BMARK_PROBLEMS")
  strip.(split(ENV["NLPMODELSIPOPT_BMARK_PROBLEMS"], ","))
else
  DEFAULT_PROBLEMS
end

SUITE["ipopt"] = BenchmarkGroup()
for prob ∈ PROBLEMS
  # decoding SIF files is expensive, so each model is created once and reused across samples
  model = CUTEstModel(prob)
  SUITE["ipopt"][prob] = @benchmarkable ipopt($model, print_level = 0)
end
