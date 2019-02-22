module NLPModelsIpopt

export ipopt

using NLPModels, Ipopt

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

  cpu_time_lines = filter(line->occursin("CPU secs", line), ipopt_output)
  Δt = 0.0
  for line in cpu_time_lines
    Δt += Meta.parse(split(line, "=")[2])
  end
  if print_output
    println(join(ipopt_output, "\n"))
  end
  x  = problem.x
  c  = problem.g
  λ  = problem.mult_g
  zL = problem.mult_x_L
  zU = problem.mult_x_U
  f  = problem.obj_val
  return x, f, c, λ, zL, zU, Ipopt.ApplicationReturnStatus[status], Δt
end

end # module
