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
    IpoptSolver(nlp)

Returns an `IpoptSolver` structure to solve the problem `nlp` with `ipopt`.
"""
mutable struct IpoptSolver{F, G, GF, JG, H, I} <: AbstractOptimizationSolver
  problem::IpoptProblem{F, G, GF, JG, H, I}
end

function define_intermediate_callback(stats::GenericExecutionStats, callback)
end

function IpoptSolver(
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats = GenericExecutionStats(nlp);
  callback = nothing,
)
  @assert get_grad_available(nlp.meta) && (get_ncon(nlp.meta) == 0 || get_jac_available(nlp.meta))
  eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = set_callbacks(nlp)
  if callback = nothing
      problem = CreateIpoptProblem(
        get_nvar(nlp.meta),
        get_lvar(nlp.meta),
        get_uvar(nlp.meta),
        get_ncon(nlp.meta),
        get_lcon(nlp.meta),
        get_ucon(nlp.meta),
        get_nnzj(nlp.meta),
        get_nnzh(nlp.meta),
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
      )
  else
      function solver_callback(
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
        args...
      )
        set_residuals!(stats, inf_pr, inf_du)
        set_iter!(stats, Int(iter_count))
        return callback(
          alg_mod,
          iter_count,
          obj_value,
          inf_pr,
          inf_du,
          mu,
          d_norm,
          regularization_size,
          alpha_du,
          alpha_pr,
          ls_trials,
          args...,
        )
      end
      problem = CreateIpoptProblem(
        get_nvar(nlp.meta),
        get_lvar(nlp.meta),
        get_uvar(nlp.meta),
        get_ncon(nlp.meta),
        get_lcon(nlp.meta),
        get_ucon(nlp.meta),
        get_nnzj(nlp.meta),
        get_nnzh(nlp.meta),
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
        solver_callback
      )
  end

  return IpoptSolver(problem)
end

"""
    solver = reset!(solver::IpoptSolver, nlp::AbstractNLPModel)

Reset the `solver` with the new model `nlp`.

If `nlp` has different bounds on the variables/constraints or a different number of nonzeros elements in the Jacobian/Hessian, then you need to create a new `IpoptSolver`.
"""
function SolverCore.reset!(solver::IpoptSolver, nlp::AbstractNLPModel)
  problem = solver.problem
  @assert get_nvar(nlp.meta) == problem.n
  @assert get_ncon(nlp.meta) == problem.m

  problem.obj_val = Inf
  problem.status = -1
  problem.x .= get_x0(nlp.meta)
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

function SolverCore.reset!(solver::IpoptSolver)
  problem = solver.problem

  problem.obj_val = Inf
  problem.status = -1  # Use -1 to indicate not solved yet
  problem.intermediate = nothing

  return solver
end

"""
    set_callbacks(nlp::AbstractNLPModel)

Return the set of functions needed to instantiate an `IpoptProblem`.
"""
function set_callbacks(nlp::AbstractNLPModel)
  eval_f(x) = obj(nlp, x)
  eval_g(x, g) = get_ncon(nlp.meta) > 0 ? cons!(nlp, x, g) : zeros(0)
  eval_grad_f(x, g) = grad!(nlp, x, g)
  eval_jac_g(x, rows::Vector{Int32}, cols::Vector{Int32}, values) = begin
    get_ncon(nlp.meta) == 0 && return
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
      if get_ncon(nlp.meta) > 0
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

Solves the `NLPModel` problem `nlp` using `Ipopt`.

For advanced usage, first define a `IpoptSolver` to preallocate the memory used in the algorithm, and then call `solve!`:
    solver = IpoptSolver(nlp)
    solve!(solver, nlp; kwargs...)
    solve!(solver, nlp, stats; kwargs...)

# Optional keyword arguments
* `x0`: a vector of size `get_nvar(nlp)` to specify an initial primal guess
* `y0`: a vector of size `get_ncon(nlp)` to specify an initial dual guess for the general constraints
* `zL`: a vector of size `get_nvar(nlp)` to specify initial multipliers for the lower bound constraints
* `zU`: a vector of size `get_nvar(nlp)` to specify initial multipliers for the upper bound constraints

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
function ipopt(nlp::AbstractNLPModel; callback = nothing, kwargs...)
  stats = GenericExecutionStats(nlp)
  solver = IpoptSolver(nlp, stats; callback = callback)
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
function ipopt(ff_nls::FeasibilityFormNLS; callback = nothing, kwargs...)
  stats = GenericExecutionStats(ff_nls)
  solver = IpoptSolver(ff_nls, stats; callback = callback)
  stats = solve!(solver, ff_nls, stats; kwargs...)

  return stats
end

function ipopt(nls::AbstractNLSModel; kwargs...)
  ff_nls = FeasibilityFormNLS(nls)
  stats = ipopt(ff_nls; kwargs...)

  stats.solution =
    length(stats.solution) >= nls.meta.nvar ? stats.solution[1:nls.meta.nvar] : stats.solution
  stats.multipliers_L =
    length(stats.multipliers_L) >= nls.meta.nvar ? stats.multipliers_L[1:nls.meta.nvar] :
    stats.multipliers_L
  stats.multipliers_U =
    length(stats.multipliers_U) >= nls.meta.nvar ? stats.multipliers_U[1:nls.meta.nvar] :
    stats.multipliers_U
  stats.multipliers =
    length(stats.multipliers) >= nls.meta.ncon ? stats.multipliers[(end - nls.meta.ncon + 1):end] :
    stats.multipliers
  return stats
end

function SolverCore.solve!(
  solver::IpoptSolver,
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats;
  kwargs...,
)
  problem = solver.problem
  SolverCore.reset!(stats)
  kwargs = Dict(kwargs)

  # Use L-BFGS if the sparse hessian of the Lagrangian is not available
  if !get_hess_available(nlp.meta)
    AddIpoptStrOption(problem, "hessian_approximation", "limited-memory")
    AddIpoptStrOption(problem, "limited_memory_update_type", "bfgs")
    AddIpoptIntOption(problem, "limited_memory_max_history", 6)
  end

  # see if user wants to warm start from an initial primal-dual guess
  if all(k ∈ keys(kwargs) for k ∈ [:x0, :y0, :zL0, :zU0])
    AddIpoptStrOption(problem, "warm_start_init_point", "yes")
    pop!(kwargs, :warm_start_init_point, nothing)  # in case the user passed this option
  end
  if :x0 ∈ keys(kwargs)
    problem.x = Vector{Float64}(kwargs[:x0])
    pop!(kwargs, :x0)
  else
    problem.x = Vector{Float64}(get_x0(nlp.meta))
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
  for (k, v) in kwargs
    if typeof(v) <: Integer
      AddIpoptIntOption(problem, string(k), v)
    elseif typeof(v) <: Real
      AddIpoptNumOption(problem, string(k), v)
    elseif typeof(v) <: String
      AddIpoptStrOption(problem, string(k), v)
    else
      @warn "$k does not seem to be a valid Ipopt option."
    end
  end

  if !get_minimize(nlp.meta)
    AddIpoptNumOption(problem, "obj_scaling_factor", -1.0)
  end


  real_time = time()
  status = IpoptSolve(problem)
  real_time = time() - real_time

  set_status!(stats, get(ipopt_statuses, status, :unknown))
  set_solution!(stats, problem.x)
  set_objective!(stats, problem.obj_val)
  set_constraint_multipliers!(stats, problem.mult_g)
  if has_bounds(nlp.meta)
    set_bounds_multipliers!(stats, problem.mult_x_L, problem.mult_x_U)
  end
  set_solver_specific!(stats, :internal_msg, ipopt_internal_statuses[status])
  set_time!(stats, real_time)

  stats
end

end # module
