module NLPModelsIpopt

export ipopt, IpoptSolver, reset!, solve!

using NLPModels, Ipopt, SolverCore
using NLPModelsModifiers: FeasibilityFormNLS

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

const ipopt_internal_statuses = Dict(
  0 => :Solve_Succeeded,
  1 => :Solved_To_Acceptable_Level,
  2 => :Infeasible_Problem_Detected,
  3 => :Search_Direction_Becomes_Too_Small,
  4 => :Diverging_Iterates,
  5 => :User_Requested_Stop,
  6 => :Feasible_Point_Found,
  -1 => :Maximum_Iterations_Exceeded,
  -2 => :Restoration_Failed,
  -3 => :Error_In_Step_Computation,
  -4 => :Maximum_CpuTime_Exceeded,
  -5 => :Maximum_WallTime_Exceeded,
  -10 => :Not_Enough_Degrees_Of_Freedom,
  -11 => :Invalid_Problem_Definition,
  -12 => :Invalid_Option,
  -13 => :Invalid_Number_Detected,
  -100 => :Unrecoverable_Exception,
  -101 => :NonIpopt_Exception_Thrown,
  -102 => :Insufficient_Memory,
  -199 => :Internal_Error,
)

"""
    IpoptSolver(nlp; kwargs...,)

Returns an `IpoptSolver` structure to solve the problem `nlp` with `ipopt`.
"""
mutable struct IpoptSolver <: AbstractOptimizationSolver
  problem::IpoptProblem
end

function IpoptSolver(nlp::AbstractNLPModel)
  eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = set_callbacks(nlp)

  problem = CreateIpoptProblem(
    nlp.meta.nvar,
    nlp.meta.lvar,
    nlp.meta.uvar,
    nlp.meta.ncon,
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
  return IpoptSolver(problem)
end

"""
    solver = reset!(solver::IpoptSolver, nlp::AbstractNLPModel)

Reset the `solver` with the new model `nlp`.

If `nlp` has different bounds on the variables/constraints or a different number of nonzeros elements in the Jacobian/Hessian, then you need to create a new `IpoptSolver`.
"""
function SolverCore.reset!(solver::IpoptSolver, nlp::AbstractNLPModel)
  problem = solver.problem
  @assert nlp.meta.nvar == problem.n
  @assert nlp.meta.ncon == problem.m

  problem.obj_val = 0.0
  problem.status = 0
  problem.x .= nlp.meta.x0
  eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = set_callbacks(nlp)
  problem.eval_f = eval_f
  problem.eval_g = eval_g
  problem.eval_grad_f = eval_grad_f
  problem.eval_jac_g = eval_jac_g
  problem.eval_h = eval_h
  problem.intermediate = nothing

  # TODO: reset problem.ipopt_problem
  return problem
end

"""
    set_callbacks(nlp::AbstractNLPModel)

Return the set of functions needed to instantiate an `IpoptProblem`.
"""
function set_callbacks(nlp::AbstractNLPModel)
  eval_f(x) = obj(nlp, x)
  eval_g(x, g) = nlp.meta.ncon > 0 ? cons!(nlp, x, g) : zeros(0)
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

  return eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
end

"""
    output = ipopt(nlp; kwargs...)

Solves the `NLPModel` problem `nlp` using `IpOpt`.

For advanced usage, first define a `IpoptSolver` to preallocate the memory used in the algorithm, and then call `solve!`:
    solver = IpoptSolver(nlp)
    solve!(solver, nlp; kwargs...)
    solve!(solver, nlp, stats; kwargs...)

# Optional keyword arguments
* `x0`: a vector of size `nlp.meta.nvar` to specify an initial primal guess
* `y0`: a vector of size `nlp.meta.ncon` to specify an initial dual guess for the general constraints
* `zL`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the lower bound constraints
* `zU`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the upper bound constraints

All other keyword arguments will be passed to Ipopt as an option.
See [https://coin-or.github.io/Ipopt/OPTIONS.html](https://coin-or.github.io/Ipopt/OPTIONS.html) for the list of options accepted.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# Examples
```
using NLPModelsIpopt, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
stats = ipopt(nlp, print_level = 0)
```

```
using NLPModelsIpopt, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
solver = IpoptSolver(nlp);
stats = solve!(solver, nlp, print_level = 0)
```
"""
function ipopt(nlp::AbstractNLPModel; kwargs...)
  solver = IpoptSolver(nlp)
  stats = GenericExecutionStats(nlp)
  return solve!(solver, nlp, stats; kwargs...)
end

"""
    ipopt(nls::AbstractNLSModel; kwargs...)

Solve the least-squares problem `nls` using `IPOPT` by moving the nonlinear residual to the constraints.

# Arguments
- `nls::AbstractNLSModel`: The least-squares problem to solve.

For advanced usage, first define an `IpoptSolver` to preallocate the memory used in the algorithm, and then call `solve!`:
    solver = IpoptSolver(nls)
    solve!(solver, nls; kwargs...)

# Examples
```julia
using NLPModelsIpopt, ADNLPModels
nls = ADNLSModel(x -> [x[1] - 1, x[2] - 2], [0.0, 0.0], 2)
stats = ipopt(nls, print_level = 0)
```
"""
function ipopt(ff_nls::FeasibilityFormNLS; kwargs...)
  solver = IpoptSolver(ff_nls)
  stats = GenericExecutionStats(ff_nls)
  stats = solve!(solver, ff_nls, stats; kwargs...)

  return stats
end

function ipopt(nls::AbstractNLSModel; kwargs...)
  ff_nls = FeasibilityFormNLS(nls)
  stats = ipopt(ff_nls; kwargs...)
  # Only keep the original variables in the solution
  stats.solution = stats.solution[1:nls.meta.nvar]
  return stats
end

function SolverCore.solve!(
  solver::IpoptSolver,
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats;
  callback = (args...) -> true,
  kwargs...,
)
  problem = solver.problem
  SolverCore.reset!(stats)
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
  SetIntermediateCallback(problem, callback)

  real_time = time()
  status = IpoptSolve(problem)
  real_time = time() - real_time

  set_status!(stats, get(ipopt_statuses, status, :unknown))
  set_solution!(stats, problem.x)
  set_objective!(stats, problem.obj_val)
  set_constraint_multipliers!(stats, problem.mult_g)
  if has_bounds(nlp)
    set_bounds_multipliers!(stats, problem.mult_x_L, problem.mult_x_U)
  end
  set_solver_specific!(stats, :internal_msg, ipopt_internal_statuses[status])
  set_solver_specific!(stats, :real_time, real_time)

  try
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
    set_residuals!(stats, primal_feas, dual_feas)
    set_iter!(stats, iter)
    set_time!(stats, Δt)
  catch e
    @warn("could not parse Ipopt log file. $e")
    stats.primal_residual_reliable = false
    stats.dual_residual_reliable = false
    stats.iter_reliable = false
    stats.time_reliable = false
  end

  stats
end

end # module
