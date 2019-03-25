using Documenter, NLPModelsIpopt

makedocs(
  modules = [NLPModelsIpopt],
  doctest = true,
  strict = true,
  assets = ["assets/style.css"],
  format = Documenter.HTML(
             prettyurls = get(ENV, "CI", nothing) == "true"
            ),
  sitename = "NLPModelsIpopt.jl",
  pages = Any["Home" => "index.md",
              "Tutorial" => "tutorial.md",
              "Reference" => "reference.md"]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl.git",
  target = "build",
  devbranch = "master"
)
