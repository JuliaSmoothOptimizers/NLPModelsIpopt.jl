## Example: Using L-BFGS (limited-memory) Hessian approximation with Ipopt

You can call Ipopt with the L-BFGS Hessian approximation by passing the following options:

```julia
stats_ipopt = ipopt(nlp,
  hessian_approximation="limited-memory",
  limited_memory_max_history=10)
```

This will use the L-BFGS method for Hessian approximation with a history size of 10.

References:
- [Ipopt.jl Manual: Hessian approximation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_Hessian_Approximation)
- [MadNLP.jl Issue #484](https://github.com/MadNLP/MadNLP.jl/issues/484)
# Tutorial

NLPModelsIpopt is a thin IPOPT wrapper for NLPModels. In this tutorial we show examples of problems created with NLPModels and solved with Ipopt.

```@contents
Pages = ["tutorial.md"]
```

## Simple problems

Calling Ipopt is simple:

```@docs
ipopt
```

Let's create an NLPModel for the Rosenbrock function
```math
f(x) = (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2
```
and solve it with Ipopt:
```@example ex1
using ADNLPModels, NLPModels, NLPModelsIpopt

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
stats = ipopt(nlp)
print(stats)
```

For comparison, we present the same problem and output using JuMP:
```@example ex2
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
optimize!(model)
```

Here is an example with a constrained problem:
```@example ex1
n = 10
x0 = ones(n)
x0[1:2:end] .= -1.2
lcon = ucon = zeros(n-2)
nlp = ADNLPModel(x -> sum((x[i] - 1)^2 + 100 * (x[i+1] - x[i]^2)^2 for i = 1:n-1), x0,
                 x -> [3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) +
                       4 * x[k+1] - x[k] * exp(x[k] - x[k+1]) - 3 for k = 1:n-2],
                 lcon, ucon)
stats = ipopt(nlp, print_level=0)
print(stats)
```

## Return value

The return value of `ipopt` is a `GenericExecutionStats` instance from `SolverCore`. It contains basic information on the solution returned by the solver.
In addition to the built-in fields of `GenericExecutionStats`, we store the detailed Ipopt output message inside `solver_specific[:internal_msg]`.

Here is an example using the constrained problem solve:
```@example ex1
stats.solver_specific[:internal_msg]
```

## Monitoring optimization with callbacks

You can monitor the optimization process using a callback function. The callback allows you to access the current iterate and constraint violations at each iteration, which is useful for custom stopping criteria, logging, or real-time analysis.

### Callback parameters

The callback function receives the following parameters from Ipopt:

- `alg_mod`: algorithm mode (0 = regular, 1 = restoration phase)
- `iter_count`: current iteration number
- `obj_value`: current objective function value
- `inf_pr`: primal infeasibility (constraint violation)
- `inf_du`: dual infeasibility 
- `mu`: complementarity measure
- `d_norm`: norm of the primal step
- `regularization_size`: size of regularization
- `alpha_du`: step size for dual variables
- `alpha_pr`: step size for primal variables  
- `ls_trials`: number of line search trials

### Example usage

Here's a complete example showing how to use callbacks to monitor the optimization:

```@example ex4
using ADNLPModels, NLPModelsIpopt

function my_callback(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, args...)
    # Log iteration information (these are the standard parameters passed by Ipopt)
    println("Iteration $iter_count:")
    println("  Objective value = ", obj_value)
    println("  Primal infeasibility = ", inf_pr)
    println("  Dual infeasibility = ", inf_du)
    println("  Complementarity = ", mu)
    
    # Return true to continue, false to stop
    return iter_count < 5  # Stop after 5 iterations for this example
end
nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
stats = ipopt(nlp, callback = my_callback, print_level = 0)
```

You can also use callbacks with the advanced solver interface:
```@example ex4
# Advanced usage with IpoptSolver
solver = IpoptSolver(nlp)
stats = solve!(solver, nlp, callback = my_callback, print_level = 0)
```

### Custom stopping criteria

Callbacks are particularly useful for implementing custom stopping criteria:

```@example ex4
function custom_stopping_callback(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, args...)
    # Custom stopping criterion: stop if objective gets close to optimum
    if obj_value < 0.01
        println("Custom stopping criterion met at iteration $iter_count")
        return false  # Stop optimization
    end
    
    return true  # Continue optimization
end

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
stats = ipopt(nlp, callback = custom_stopping_callback, print_level = 0)
```

## Manual input

In this section, we work through an example where we specify the problem and its derivatives manually. For this, we need to implement the following `NLPModel` API methods:
- `obj(nlp, x)`: evaluate the objective value at `x`;
- `grad!(nlp, x, g)`: evaluate the objective gradient at `x`;
- `cons!(nlp, x, c)`: evaluate the vector of constraints, if any;
- `jac_structure!(nlp, rows, cols)`: fill `rows` and `cols` with the spartity structure of the Jacobian, if the problem is constrained;
- `jac_coord!(nlp, x, vals)`: fill `vals` with the Jacobian values corresponding to the sparsity structure returned by `jac_structure!()`;
- `hess_structure!(nlp, rows, cols)`: fill `rows` and `cols` with the spartity structure of the lower triangle of the Hessian of the Lagrangian;
- `hess_coord!(nlp, x, y, vals; obj_weight=1.0)`: fill `vals` with the values of the Hessian of the Lagrangian corresponding to the sparsity structure returned by `hess_structure!()`, where `obj_weight` is the weight assigned to the objective, and `y` is the vector of multipliers.

The model that we implement is a logistic regression model. We consider the model ``h(\beta; x) = (1 + e^{-\beta^Tx})^{-1}``, and the loss function
```math
\ell(\beta) = -\sum_{i = 1}^m y_i \ln h(\beta; x_i) + (1 - y_i) \ln(1 - h(\beta; x_i))
```
with regularization ``\lambda \|\beta\|^2 / 2``.

```@example ex3
using DataFrames, LinearAlgebra, NLPModels, NLPModelsIpopt, Random

mutable struct LogisticRegression <: AbstractNLPModel{Float64, Vector{Float64}}
  X :: Matrix
  y :: Vector
  λ :: Real
  meta :: NLPModelMeta{Float64, Vector{Float64}} # required by AbstractNLPModel
  counters :: Counters # required by AbstractNLPModel
end

function LogisticRegression(X, y, λ = 0.0)
  m, n = size(X)
  meta = NLPModelMeta(n, name="LogisticRegression", nnzh=div(n * (n+1), 2) + n) # nnzh is the length of the coordinates vectors
  return LogisticRegression(X, y, λ, meta, Counters())
end

function NLPModels.obj(nlp :: LogisticRegression, β::AbstractVector)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  return -sum(nlp.y .* log.(hβ .+ 1e-8) .+ (1 .- nlp.y) .* log.(1 .- hβ .+ 1e-8)) + nlp.λ * dot(β, β) / 2
end

function NLPModels.grad!(nlp :: LogisticRegression, β::AbstractVector, g::AbstractVector)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  g .= nlp.X' * (hβ .- nlp.y) + nlp.λ * β
end

function NLPModels.hess_structure!(nlp :: LogisticRegression, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= [getindex.(I, 1); 1:n]
  cols[1 : nlp.meta.nnzh] .= [getindex.(I, 2); 1:n]
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: LogisticRegression, β::AbstractVector, vals::AbstractVector; obj_weight=1.0, y=Float64[])
  n, m = nlp.meta.nvar, length(nlp.y)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  fill!(vals, 0.0)
  for k = 1:m
    hk = hβ[k]
    p = 1
    for j = 1:n, i = j:n
      vals[p] += obj_weight * hk * (1 - hk) * nlp.X[k,i] * nlp.X[k,j]
      p += 1
    end
  end
  vals[nlp.meta.nnzh+1:end] .= nlp.λ * obj_weight
  return vals
end

Random.seed!(0)

# Training set
m = 1000
df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
df.buy = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)

X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
y = df.buy

λ = 1.0e-2
nlp = LogisticRegression(X, y, λ)
stats = ipopt(nlp, print_level=0)
β = stats.solution

# Test set - same generation method
m = 100
df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
df.buy = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)

X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
hβ = 1 ./ (1 .+ exp.(-X * β))
ypred = hβ .> 0.5

acc = count(df.buy .== ypred) / m
println("acc = $acc")
```

```@setup ex3
using Plots
gr()


f(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])
P = findall(df.buy .== true)
scatter(df.age[P], df.salary[P], c=:blue, m=:square)
P = findall(df.buy .== false)
scatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)
contour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])
png("ex3")
```

```
using Plots
gr()

f(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])
P = findall(df.buy .== true)
scatter(df.age[P], df.salary[P], c=:blue, m=:square)
P = findall(df.buy .== false)
scatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)
contour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])
```

![](ex3.png)
