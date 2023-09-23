ENV["GKSwstype"] = "100"
using Documenter, NLPModelsIpopt

makedocs(
  modules = [NLPModelsIpopt],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
  ),
  sitename = "NLPModelsIpopt.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl.git",
  devbranch = "main",
  push_preview = true,
)
