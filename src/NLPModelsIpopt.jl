module NLPModelsIpopt

export ipopt

using NLPModels, Ipopt

"""`output = ipopt(nlp)`

Solves the `NLPModel` problem `nlp` using `IpOpt`.
"""
function ipopt(nlp :: AbstractNLPModel;
               callback :: Union{Function,Nothing} = nothing,
               kwargs...)
  n, m = nlp.meta.nvar, nlp.meta.ncon
  eval_f(x) = obj(nlp, x)
  eval_g(x, g) = m > 0 ? cons!(nlp, x, g) : zeros(0)
  eval_grad_f(x, g) = grad!(nlp, x, g)
  eval_jac_g(x, mode, rows::Vector{Int32}, cols::Vector{Int32}, values) = begin
    nlp.meta.ncon == 0 && return
    if mode == :Structure
      x = [-(-1.0)^i for i = 1:nlp.meta.nvar]
      I, J, _ = jac_coord(nlp, x)
      rows .= I
      cols .= J
    else
      I, J, V = jac_coord(nlp, x)
      values .= V
    end
  end
  eval_h(x, mode, rows::Vector{Int32}, cols::Vector{Int32}, σ, λ, values) = begin
    if mode == :Structure
      x = [-(-1.0)^i for i = 1:nlp.meta.nvar]
      λ = [-(-1.0)^i for i = 1:nlp.meta.ncon]
      I, J, _ = hess_coord(nlp, x, obj_weight=1.0, y=λ)
      rows .= I
      cols .= J
    else
      if nlp.meta.ncon > 0
        I, J, V = hess_coord(nlp, x, obj_weight=σ, y=λ)
      else
        I, J, V = hess_coord(nlp, x, obj_weight=σ)
      end
      values .= V
    end
  end

  problem = createProblem(n, nlp.meta.lvar, nlp.meta.uvar,
                          m, nlp.meta.lcon, nlp.meta.ucon,
                          nlp.meta.nnzj, nlp.meta.nnzh,
                          eval_f, eval_g, eval_grad_f,
                          eval_jac_g, eval_h)
  problem.x = copy(nlp.meta.x0)

  # Options
  for (k,v) in kwargs
    addOption(problem, string(k), v)
  end

  # Callback
  callback === nothing || setIntermediateCallback(problem, callback)

  status = solveProblem(problem)
  x  = problem.x
  c  = problem.g
  λ  = problem.mult_g
  zL = problem.mult_x_L
  zU = problem.mult_x_U
  f  = problem.obj_val
  return x, f, c, λ, zL, zU, Ipopt.ApplicationReturnStatus[status]
end

end # module
