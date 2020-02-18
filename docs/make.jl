using Documenter, NLPModelsIpopt

makedocs(
  modules = [NLPModelsIpopt],
  doctest = true,
  strict = true,
  format = Documenter.HTML(
             prettyurls = get(ENV, "CI", nothing) == "true",
             assets = ["assets/style.css"],
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
