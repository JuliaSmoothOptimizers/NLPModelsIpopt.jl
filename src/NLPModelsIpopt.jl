module NLPModelsIpopt

export ipopt

using NLPModels, Ipopt, SolverCore

const ipopt_statuses = Dict(
  0 => :first_order,
  1 => :acceptable,
  2 => :infeasible,
  3 => :small_step,
  #4 => Diverging iterates
  5 => :user,
  #6 => Feasible point found
  -1 => :max_iter,
  #-2 => Restoration failed
  #-3 => Error in step computation
  -4 => :max_time, # Maximum cputime exceeded
  -5 => :max_time, # Maximum walltime exceeded
  #-10 => Not enough degress of freedom
  #-11 => Invalid problem definition
  #-12 => Invalid option
  #-13 => Invalid number detected
  -100 => :exception, # Unrecoverable exception
  -101 => :exception, # NonIpopt exception thrown
  -102 => :exception, # Insufficient memory
  -199 => :exception, # Internal error
)

"""`output = ipopt(nlp; kwargs...)`

Solves the `NLPModel` problem `nlp` using `IpOpt`.

# Optional keyword arguments
* `x0`: a vector of size `nlp.meta.nvar` to specify an initial primal guess
* `y0`: a vector of size `nlp.meta.ncon` to specify an initial dual guess for the general constraints
* `zL`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the lower bound constraints
* `zU`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the upper bound constraints

All other keyword arguments will be passed to IpOpt as an option.
See https://coin-or.github.io/Ipopt/OPTIONS.html for the list of options accepted.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# Examples
```
using NLPModelsIpopt, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
stats = ipopt(nlp, print_level = 0)
```
"""
function ipopt(nlp::AbstractNLPModel; callback::Union{Function, Nothing} = nothing, kwargs...)
  n, m = nlp.meta.nvar, nlp.meta.ncon

  eval_f(x) = obj(nlp, x)
  eval_g(x, g) = m > 0 ? cons!(nlp, x, g) : zeros(0)
  eval_grad_f(x, g) = grad!(nlp, x, g)
  eval_jac_g(x, rows::Vector{Int32}, cols::Vector{Int32}, values) = begin
    nlp.meta.ncon == 0 && return
    if values == nothing
      jac_structure!(nlp, rows, cols)
    else
      jac_coord!(nlp, x, values)
    end
  end
  eval_h(x, rows::Vector{Int32}, cols::Vector{Int32}, σ, λ, values) = begin
    if values == nothing
      hess_structure!(nlp, rows, cols)
    else
      if nlp.meta.ncon > 0
        hess_coord!(nlp, x, λ, values, obj_weight = σ)
      else
        hess_coord!(nlp, x, values, obj_weight = σ)
      end
    end
  end

  problem = CreateIpoptProblem(
    n,
    nlp.meta.lvar,
    nlp.meta.uvar,
    m,
    nlp.meta.lcon,
    nlp.meta.ucon,
    nlp.meta.nnzj,
    nlp.meta.nnzh,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h,
  )

  kwargs = Dict(kwargs)

  # see if user wants to warm start from an initial primal-dual guess
  if all(k ∈ keys(kwargs) for k ∈ [:x0, :y0, :zL0, :zU0])
    AddIpoptStrOption(problem, "warm_start_init_point", "yes")
    pop!(kwargs, :warm_start_init_point, nothing)  # in case the user passed this option
  end
  if :x0 ∈ keys(kwargs)
    problem.x = Vector{Float64}(kwargs[:x0])
    pop!(kwargs, :x0)
  else
    problem.x = Vector{Float64}(nlp.meta.x0)
  end
  if :y0 ∈ keys(kwargs)
    problem.mult_g = Vector{Float64}(kwargs[:y0])
    pop!(kwargs, :y0)
  end
  if :zL0 ∈ keys(kwargs)
    problem.mult_x_L = Vector{Float64}(kwargs[:zL0])
    pop!(kwargs, :zL0)
  end
  if :zU0 ∈ keys(kwargs)
    problem.mult_x_U = Vector{Float64}(kwargs[:zU0])
    pop!(kwargs, :zU0)
  end

  # pass options to IPOPT
  # make sure IPOPT logs to file so we can grep time, residuals and number of iterations
  ipopt_log_to_file = false
  ipopt_file_log_level = 3
  local ipopt_log_file
  for (k, v) in kwargs
    if k == :output_file
      ipopt_log_file = v
      ipopt_log_to_file = true
    elseif k == :file_print_level
      ipopt_file_log_level = v
    elseif typeof(v) <: Integer
      AddIpoptIntOption(problem, string(k), v)
    elseif typeof(v) <: Real
      AddIpoptNumOption(problem, string(k), v)
    elseif typeof(v) <: String
      AddIpoptStrOption(problem, string(k), v)
    else
      @warn "$k does not seem to be a valid Ipopt option."
    end
  end

  if !nlp.meta.minimize
    AddIpoptNumOption(problem, "obj_scaling_factor", -1.0)
  end

  if ipopt_log_to_file
    0 < ipopt_file_log_level < 3 && @warn(
      "`file_print_level` should be 0 or ≥ 3 for IPOPT to report elapsed time, final residuals and number of iterations"
    )
  else
    # log to file anyways to parse the output
    ipopt_log_file = tempname()
    # make sure the user didn't specify a file log level without a file name
    0 < ipopt_file_log_level < 3 && (ipopt_file_log_level = 3)
  end

  AddIpoptStrOption(problem, "output_file", ipopt_log_file)
  AddIpoptIntOption(problem, "file_print_level", ipopt_file_log_level)

  # Callback
  callback === nothing || SetIntermediateCallback(problem, callback)

  real_time = time()
  status = IpoptSolve(problem)
  real_time = time() - real_time
  ipopt_output = readlines(ipopt_log_file)

  Δt = 0.0
  dual_feas = primal_feas = Inf
  iter = -1
  for line in ipopt_output
    if occursin("Total seconds", line)
      Δt += Meta.parse(split(line, "=")[2])
    elseif occursin("Dual infeasibility", line)
      dual_feas = Meta.parse(split(line)[4])
    elseif occursin("Constraint violation", line)
      primal_feas = Meta.parse(split(line)[4])
    elseif occursin("Number of Iterations....", line)
      iter = Meta.parse(split(line)[4])
    end
  end

  return GenericExecutionStats(
    get(ipopt_statuses, status, :unknown),
    nlp,
    solution = problem.x,
    objective = problem.obj_val,
    dual_feas = dual_feas,
    iter = iter,
    primal_feas = primal_feas,
    elapsed_time = Δt,
    multipliers = problem.mult_g,
    multipliers_L = problem.mult_x_L,
    multipliers_U = problem.mult_x_U,
    solver_specific = Dict(:internal_msg => Ipopt._STATUS_CODES[status], :real_time => real_time),
  )
end

end # module
