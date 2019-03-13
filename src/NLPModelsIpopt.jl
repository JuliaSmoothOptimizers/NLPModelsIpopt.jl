module NLPModelsIpopt

export ipopt

using NLPModels, Ipopt, SolverTools

const ipopt_statuses = Dict(0 => :first_order,
                            1 => :first_order,
                            2 => :infeasible,
                            3 => :small_step,
                            #4 => Diverging iterates
                            #5 => User requestep stop
                            #6 => Feasible point found
                            -1 => :max_iter,
                            #-2 => Restoration failed
                            #-3 => Error in step computation
                            -4 => :max_time,
                            #-10 => Not enough degress of freedom
                            #-11 => Invalid problem definition
                            #-12 => Invalid option
                            #-13 => Invalid number detected
                            -100 => :exception,
                            -101 => :exception,
                            -102 => :exception,
                            -199 => :exception)

"""`output = ipopt(nlp)`

Solves the `NLPModel` problem `nlp` using `IpOpt`.
"""
function ipopt(nlp :: AbstractNLPModel;
               callback :: Union{Function,Nothing} = nothing,
               ignore_time :: Bool = false,
               kwargs...)
  n, m = nlp.meta.nvar, nlp.meta.ncon
  local jrows, jcols, hrows, hcols

  eval_f(x) = obj(nlp, x)
  eval_g(x, g) = m > 0 ? cons!(nlp, x, g) : zeros(0)
  eval_grad_f(x, g) = grad!(nlp, x, g)
  eval_jac_g(x, mode, rows::Vector{Int32}, cols::Vector{Int32}, values) = begin
    nlp.meta.ncon == 0 && return
    if mode == :Structure
      jrows, jcols = jac_structure(nlp)
      rows .= jrows
      cols .= jcols
    else
      jac_coord!(nlp, x, jrows, jcols, values)
    end
  end
  eval_h(x, mode, rows::Vector{Int32}, cols::Vector{Int32}, σ, λ, values) = begin
    if mode == :Structure
      hrows, hcols = hess_structure(nlp)
      rows .= hrows
      cols .= hcols
    else
      if nlp.meta.ncon > 0
        hess_coord!(nlp, x, hrows, hcols, values, obj_weight=σ, y=λ)
      else
        hess_coord!(nlp, x, hrows, hcols, values, obj_weight=σ)
      end
    end
  end

  problem = createProblem(n, nlp.meta.lvar, nlp.meta.uvar,
                          m, nlp.meta.lcon, nlp.meta.ucon,
                          nlp.meta.nnzj, nlp.meta.nnzh,
                          eval_f, eval_g, eval_grad_f,
                          eval_jac_g, eval_h)
  problem.x = copy(nlp.meta.x0)

  print_output = true

  # Options
  for (k,v) in kwargs
    if ignore_time || k != :print_level || v ≥ 3
      addOption(problem, string(k), v)
    else
      if v > 0
        @warn("`print_level` should be 0 or ≥ 3 to get the elapsed time, if you don't care about the elapsed time, pass `ignore_time=true`")
      end
      print_output = false
      addOption(problem, "print_level", 3)
    end
  end

  # Callback
  callback === nothing || setIntermediateCallback(problem, callback)

  tmpfile = tempname()
  local status
  open(tmpfile, "w") do io
    redirect_stdout(io) do
      status = solveProblem(problem)
    end
  end
  ipopt_output = readlines(tmpfile)

  Δt = 0.0
  dual_feas = primal_feas = Inf
  for line in ipopt_output
    if occursin("CPU secs", line)
      Δt += Meta.parse(split(line, "=")[2])
    elseif occursin("Dual infeasibility", line)
      dual_feas = Meta.parse(split(line)[4])
    elseif occursin("Constraint violation", line)
      primal_feas = Meta.parse(split(line)[4])
    end
  end
  if print_output
    println(join(ipopt_output, "\n"))
  end

  return GenericExecutionStats(get(ipopt_statuses, status, :unknown), nlp, solution=problem.x,
                               objective=problem.obj_val, dual_feas=dual_feas,
                               primal_feas=primal_feas, elapsed_time=Δt,
                               solver_specific=Dict(:multipliers_con => problem.mult_g,
                                                    :multipliers_L => problem.mult_x_L,
                                                    :multipliers_U => problem.mult_x_U,
                                                    :internal_msg => Ipopt.ApplicationReturnStatus[status])
                              )
end

end # module
