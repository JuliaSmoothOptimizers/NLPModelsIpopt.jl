using BenchmarkTools

using CUTEst
using NLPModelsIpopt

const SUITE = BenchmarkGroup()

# all_sif_problems = CUTEst.select()
all_sif_problems = ("ROSENBR", "WOODS", "PENALTY1")  # to debug
for prob ∈ all_sif_problems
    SUITE[prob] = @benchmarkable ipopt(model) setup = (model = CUTEstModel($prob)) teardown = (finalize(model))
end

