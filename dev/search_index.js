var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#NLPModelsIpopt.jl-documentation-1",
    "page": "Home",
    "title": "NLPModelsIpopt.jl documentation",
    "category": "section",
    "text": "This package provides a thin IPOPT wrapper for NLPModels, using JuliaOpt/Ipopt.jl internal structures directly.Please refer to the NLPModels documentation for the API of NLPModels, if needed."
},

{
    "location": "#Install-1",
    "page": "Home",
    "title": "Install",
    "category": "section",
    "text": "Install NLPModelsIpopt.jl with the following commands.pkg> add NLPModelsIpopt"
},

{
    "location": "#Contents-1",
    "page": "Home",
    "title": "Contents",
    "category": "section",
    "text": ""
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": "NLPModelsIpopt is a thin IPOPT wrapper for NLPModels. In this tutorial we\'ll show examples of problems created with NLPModels and solved with Ipopt.Pages = [\"tutorial.md\"]"
},

{
    "location": "tutorial/#NLPModelsIpopt.ipopt",
    "page": "Tutorial",
    "title": "NLPModelsIpopt.ipopt",
    "category": "function",
    "text": "output = ipopt(nlp)\n\nSolves the NLPModel problem nlp using IpOpt.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#Simple-problems-1",
    "page": "Tutorial",
    "title": "Simple problems",
    "category": "section",
    "text": "The interface for calling Ipopt is very simple:ipoptLet\'s create an NLPModel for the Rosenbrock functionf(x) = (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2to test this interface:using NLPModels, NLPModelsIpopt\n\nnlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])\nstats = ipopt(nlp)\nprint(stats)For comparison, we present the same problem and output using the JuMP route:using JuMP, Ipopt\n\nmodel = Model(with_optimizer(Ipopt.Optimizer))\nx0 = [-1.2; 1.0]\n@variable(model, x[i=1:2], start=x0[i])\n@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)\noptimize!(model)Another example, using a constrained problemn = 10\nx0 = ones(n)\nx0[1:2:end] .= -1.2\nnlp = ADNLPModel(x -> sum((x[i] - 1)^2 + 100 * (x[i+1] - x[i]^2)^2 for i = 1:n-1), x0,\n                 c=x -> [3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) +\n                         4 * x[k+1] - x[k] * exp(x[k] - x[k+1]) - 3 for k = 1:n-2],\n                 lcon=zeros(n-2), ucon=zeros(n-2))\nstats = ipopt(nlp, print_level=0)\nprint(stats)"
},

{
    "location": "tutorial/#Output-1",
    "page": "Tutorial",
    "title": "Output",
    "category": "section",
    "text": "The output of ipopt is a GenericExecutionStats from SolverTools. It contains basic information from the solver. In addition to the built-in fields of GenericExecutionStats, we also store in solver_specific the following fields:multipliers_con: Constraints multipliers;\nmultipliers_L: Variables lower-bound multipliers;\nmultipliers_U: Variables upper-bound multipliers;\ninternal_msg: Detailed Ipopt output message.stats.solver_specific[:internal_msg]"
},

{
    "location": "tutorial/#Manual-input-1",
    "page": "Tutorial",
    "title": "Manual input",
    "category": "section",
    "text": "This is an example where we specify the problem and its derivatives manually. For this, we create an NLPModel, and we need to define the following API functions:obj(nlp, x): objective\ngrad!(nlp, x, g): gradient\ncons!(nlp, x, c): constraints, if any\njac_structure!(nlp, rows, cols): structure of the Jacobian, if constrained;\njac_coord!(nlp, x, rows, cols, vals): Jacobian values (the user should not attempt to access rows and cols, as Ipopt doesn\'t actually pass them);\nhess_structure!(nlp, rows, cols): structure of the lower triangle of the Hessian of the Lagrangian;\nhess_coord!(nlp, x, rows, cols, vals; obj_weight=1.0, y=[]): Hessian of the Lagrangian, where obj_weight is the weight assigned to the objective, and y is the multipliers vector (the user should not attempt to access rows and cols, as Ipopt doesn\'t actually pass them).Let\'s implement a logistic regression model. We consider the model h(beta x) = (1 + e^-beta^Tx)^-1, and the loss functionell(beta) = -sum_i = 1^m y_i ln h(beta x_i) + (1 - y_i) ln(1 - h(beta x_i))with regularization lambda beta^2  2.using DataFrames, LinearAlgebra, NLPModels, NLPModelsIpopt, Random\n\nmutable struct LogisticRegression <: AbstractNLPModel\n  X :: Matrix\n  y :: Vector\n  λ :: Real\n  meta :: NLPModelMeta # required by AbstractNLPModel\n  counters :: Counters # required by AbstractNLPModel\nend\n\nfunction LogisticRegression(X, y, λ = 0.0)\n  m, n = size(X)\n  meta = NLPModelMeta(n, name=\"LogisticRegression\", nnzh=div(n * (n+1), 2) + n) # nnzh is the length of the coordinates vectors\n  return LogisticRegression(X, y, λ, meta, Counters())\nend\n\nfunction NLPModels.obj(nlp :: LogisticRegression, β::AbstractVector)\n  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))\n  return -sum(nlp.y .* log.(hβ .+ 1e-8) .+ (1 .- nlp.y) .* log.(1 .- hβ .+ 1e-8)) + nlp.λ * dot(β, β) / 2\nend\n\nfunction NLPModels.grad!(nlp :: LogisticRegression, β::AbstractVector, g::AbstractVector)\n  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))\n  g .= nlp.X\' * (hβ .- nlp.y) + nlp.λ * β\nend\n\nfunction NLPModels.hess_structure!(nlp :: LogisticRegression, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})\n  n = nlp.meta.nvar\n  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)\n  rows[1 : nlp.meta.nnzh] .= [getindex.(I, 1); 1:n]\n  cols[1 : nlp.meta.nnzh] .= [getindex.(I, 2); 1:n]\n  return rows, cols\nend\n\nfunction NLPModels.hess_coord!(nlp :: LogisticRegression, β::AbstractVector, rows::AbstractVector{<: Integer}, cols::AbstractVector{<: Integer}, vals::AbstractVector; obj_weight=1.0, y=Float64[])\n  n, m = nlp.meta.nvar, length(nlp.y)\n  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))\n  fill!(vals, 0.0)\n  for k = 1:m\n    hk = hβ[k]\n    p = 1\n    for j = 1:n, i = j:n\n      vals[p] += obj_weight * hk * (1 - hk) * nlp.X[k,i] * nlp.X[k,j]\n      p += 1\n    end\n  end\n  vals[nlp.meta.nnzh+1:end] .= nlp.λ * obj_weight\n  return rows, cols, vals\nend\n\nRandom.seed!(0)\n\n# Training set\nm = 1000\ndf = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)\ndf[:buy] = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)\n\nX = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]\ny = df.buy\n\nλ = 1.0e-2\nnlp = LogisticRegression(X, y, λ)\nstats = ipopt(nlp, print_level=0)\nβ = stats.solution\n\n# Test set - same generation method\nm = 100\ndf = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)\ndf[:buy] = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)\n\nX = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]\nhβ = 1 ./ (1 .+ exp.(-X * β))\nypred = hβ .> 0.5\n\nacc = count(df.buy .== ypred) / m\nprintln(\"acc = $acc\")using Plots\ngr()\n\nf(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])\nP = findall(df.buy .== true)\nscatter(df.age[P], df.salary[P], c=:blue, m=:square)\nP = findall(df.buy .== false)\nscatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)\ncontour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])\npng(\"ex3\")using Plots\ngr()\n\nf(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])\nP = findall(df.buy .== true)\nscatter(df.age[P], df.salary[P], c=:blue, m=:square)\nP = findall(df.buy .== false)\nscatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)\ncontour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])(Image: )"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

]}
