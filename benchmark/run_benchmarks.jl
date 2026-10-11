# Compare the performance of the current state of the repository against a baseline.
#
# Usage:
#   julia benchmark/run_benchmarks.jl [outdir] [baseline] [script]
#
#   outdir    directory where results are written (default: benchmark_results)
#   baseline  git reference to compare against (default: main)
#   script    benchmark suite in benchmark/ to run (default: benchmarks.jl)
#
# The working tree must be clean because PkgBenchmark checks out `baseline`.
using Pkg

const bmark_dir = @__DIR__
const pkg_dir = normpath(joinpath(bmark_dir, ".."))

outdir = abspath(length(ARGS) ≥ 1 ? ARGS[1] : "benchmark_results")
baseline = length(ARGS) ≥ 2 ? ARGS[2] : "main"
script_name = length(ARGS) ≥ 3 ? ARGS[3] : "benchmarks.jl"

# The benchmark script and environment may not exist on `baseline`, so copy them to a
# temporary location that is unaffected when PkgBenchmark checks out `baseline`.
const tmp_dir = mktempdir()
const script = joinpath(tmp_dir, script_name)
cp(joinpath(bmark_dir, script_name), script)
cp(joinpath(bmark_dir, "Project.toml"), joinpath(tmp_dir, "Project.toml"))
Pkg.activate(tmp_dir)
Pkg.develop(PackageSpec(path = pkg_dir))
Pkg.instantiate()

using DataFrames
using JLD2
using PkgBenchmark
using Plots
using SolverBenchmark

commit = benchmarkpkg(pkg_dir, script = script)  # current state of repository
main = benchmarkpkg(pkg_dir, baseline, script = script)
judgement = judge(commit, main)

commit_stats = bmark_results_to_dataframes(commit)
main_stats = bmark_results_to_dataframes(main)
judgement_stats = judgement_results_to_dataframes(judgement)

mkpath(outdir)
export_markdown(joinpath(outdir, "judgement.md"), judgement)
export_markdown(joinpath(outdir, "summary.md"), judgement)
export_markdown(joinpath(outdir, "baseline.md"), main)
export_markdown(joinpath(outdir, "commit.md"), commit)

# Time performance profile next to the commit/baseline ratio of each metric per benchmark.
# Profiles of metrics that rarely change, such as memory, are flat and hard to read, so
# those metrics are only shown as ratios.
function plot_commit_vs_baseline(stats::Dict{Symbol, DataFrame})
  profile = performance_profile(stats, df -> df[!, :time], title = "time")
  commit, baseline = stats[:commit], stats[:baseline]
  ratios = plot(
    title = "commit / baseline",
    xticks = (1:nrow(commit), commit[!, :name]),
    xrotation = 45,
    legend = :outerright,
  )
  hline!(ratios, [1.0], color = :gray, linestyle = :dash, label = "")
  for metric ∈ (:time, :memory, :allocations)
    scatter!(ratios, commit[!, metric] ./ baseline[!, metric], label = string(metric))
  end
  plot(
    profile,
    ratios,
    layout = (1, 2),
    size = (1200, 450),
    margin = 5Plots.mm,
    bottom_margin = 10Plots.mm,
  )
end

for k ∈ keys(judgement_stats)
  k_stats = Dict{Symbol, DataFrame}(
    :commit => sort(commit_stats[k], :name),
    :baseline => sort(main_stats[k], :name),
  )
  save_stats(k_stats, joinpath(outdir, "commit_vs_baseline_$(k).jld2"), force = true)
  plot_commit_vs_baseline(k_stats)
  savefig(joinpath(outdir, "profiles_commit_vs_baseline_$(k).svg"))
  savefig(joinpath(outdir, "profiles_commit_vs_baseline_$(k).png"))
end

jldopen(joinpath(outdir, "commit_vs_baseline_judgement.jld2"), "w") do file
  file["jstats"] = judgement_stats
end

println("Benchmark results written to $outdir")
