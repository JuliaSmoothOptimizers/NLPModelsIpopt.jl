using ADNLPModels, NLPModelsIpopt, NLPModels, Ipopt, SolverCore, Test
using NLPModelsModifiers: FeasibilityFormNLS

@testset "Restart NLPModelsIpopt" begin
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = GenericExecutionStats(nlp)
  solver = IpoptSolver(nlp)
  stats = solve!(solver, nlp, stats, print_level = 0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.iter == 21
  @test stats.elapsed_time > 0
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8

  nlp = ADNLPModel(x -> (x[1])^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  SolverCore.reset!(solver, nlp)
  stats = solve!(solver, nlp, stats, print_level = 0)
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
  @test stats.status == :first_order
  @test stats.iter == 16
  @test stats.elapsed_time > 0
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8
end

@testset "Unit tests NLPModelsIpopt" begin
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, print_level = 0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.elapsed_time > 0
  @test stats.iter == 21
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol = 1e-12, print_level = 0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.elapsed_time > 0
  @test stats.iter == 22
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0

  # solve again from solution
  x0 = copy(stats.solution)
  stats = ipopt(nlp, x0 = x0, tol = 1e-12, print_level = 0)
  @test isapprox(stats.solution, x0, rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.iter == 0
  @test stats.elapsed_time >= 0
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0

  function callback(alg_mod, iter_count, args...)
    return iter_count < 1
  end
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol = 1e-12, callback = callback, print_level = 0)
  @test stats.status == :user
  @test stats.solver_specific[:internal_msg] == :User_Requested_Stop
  @test stats.iter == 1
  @test stats.elapsed_time > 0
  @test stats.primal_feas ≈ 0.0
  # @test stats.dual_feas ≈ 4.63

  @testset "JSO callback stops after 5 iterations" begin
    function jso_callback(nlp_in, solver_in, stats_in)
      @test typeof(nlp_in) <: AbstractNLPModel
      @test hasproperty(stats_in, :iter)
      return stats_in.iter < 5
    end
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    stats = ipopt(nlp, tol = 1e-12, callback = jso_callback, print_level = 0)
    @test stats.status == :user
    @test stats.solver_specific[:internal_msg] == :User_Requested_Stop
    @test stats.iter == 5
  end

  @testset "JSO callback can read problem and nlp" begin
    function jso_cb_problem_nlp(nlp_in, solver_in, stats_in)
      @test typeof(nlp_in) <: AbstractNLPModel
      @test length(solver_in.x) == nlp_in.meta.nvar
      if nlp_in.meta.ncon > 0
        @test length(solver_in.mult_g) == nlp_in.meta.ncon
      end
      return stats_in.iter < 3
    end
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    stats = ipopt(nlp, callback = jso_cb_problem_nlp, print_level = 0)
    @test stats.status == :user
    @test stats.iter == 3
  end

  @testset "Short Ipopt-style 3-arg callback" begin
    function short_cb(alg_mod, iter_count, obj_value)
      @test isa(alg_mod, Integer)
      @test iter_count >= 0
      @test isa(obj_value, Real)
      return iter_count < 4
    end
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    stats = ipopt(nlp, callback = short_cb, callback_style = :ipopt_short, print_level = 0)
    @test stats.status == :user
    @test stats.iter == 4
  end

  @testset "JSO callback can use solver and nlp" begin
    used_solver = Ref(false)
    used_nlp = Ref(false)
    function jso_cb(nlp_in, solver_in, stats_in)
      # Use solver.x (problem current iterate)
      @test length(solver_in.x) == nlp_in.meta.nvar
      used_solver[] = true
      # Use nlp to compute objective at current x
      _ = obj(nlp_in, solver_in.x)
      used_nlp[] = true
      return stats_in.iter < 3
    end
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    stats = ipopt(nlp, callback = jso_cb, print_level = 0)
    @test stats.status == :user
    @test used_solver[]
    @test used_nlp[]
    @test stats.iter == 3
  end

  @testset "Ipopt-style short callback (3 args)" begin
    function short_cb(alg_mod, iter_count, obj_value)
      @test isa(alg_mod, Integer)
      @test isa(iter_count, Integer)
      @test isa(obj_value, Real)
      return iter_count < 2
    end
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    stats = ipopt(nlp, callback = short_cb, callback_style = :ipopt_short, print_level = 0)
    @test stats.status == :user
    @test stats.iter == 2
  end

  nlp =
    ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2), x -> [sum(x) - 1.0], [0.0], [0.0])
  stats = ipopt(nlp, print_level = 0)
  @test isapprox(stats.solution, [-1.4; 2.4], rtol = 1e-6)
  @test stats.iter == 1
  @test stats.status == :first_order
  @test stats.elapsed_time > 0
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8

  # solve constrained problem again from solution
  x0 = copy(stats.solution)
  y0 = copy(stats.multipliers)
  # Ipopt wants zL and zU of size n, whether there are bounds or not
  zL = has_bounds(nlp) ? copy(stats.multipliers_L) : zeros(nlp.meta.nvar)
  zU = has_bounds(nlp) ? copy(stats.multipliers_U) : zeros(nlp.meta.nvar)
  stats = ipopt(nlp, x0 = x0, y0 = y0, zL0 = zL, zU0 = zU, print_level = 0)
  @test isapprox(stats.solution, x0, rtol = 1e-6)
  @test isapprox(stats.multipliers, y0, rtol = 1e-6)
  if has_bounds(nlp)
    @test isapprox(stats.multipliers_L, zL, rtol = 1e-6)
    @test isapprox(stats.multipliers_U, zU, rtol = 1e-6)
  end
  @test stats.elapsed_time >= 0
  @test stats.iter == 0
  @test stats.status == :first_order
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8

  x0, f = rand(1), x -> x[1]
  nlp = ADNLPModel(f, x0, zeros(1), ones(1), minimize = false)
  @test nlp.meta.minimize == false
  stats = ipopt(nlp, print_level = 0)
  @test isapprox(stats.solution, ones(1), rtol = 1e-6)
  @test isapprox(stats.objective, 1.0, rtol = 1e-6)
  @test isapprox(stats.multipliers_L, zeros(1), atol = 1e-6)
  @test isapprox(stats.multipliers_U, -ones(1), rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.elapsed_time > 0
  @test stats.iter in [4; 5]
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8
end

@testset "ipopt with AbstractNLSModel" begin
  nls = ADNLSModel(x -> [x[1] - 1, x[2] - 2], [0.0, 0.0], 2)
  stats = ipopt(nls, print_level = 0)
  @test isapprox(stats.solution, [1.0, 2.0], rtol = 1e-6)
  @test stats.status == :first_order
  @test stats.iter >= 0
  @test isapprox(stats.dual_feas, 0.0; atol = 1e-8)
end

@testset "Test restart with a different initial guess" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  stats = GenericExecutionStats(nlp)
  solver = IpoptSolver(nlp)

  # Solve the problem first
  stats = solve!(solver, nlp, stats, print_level = 0)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  # Change initial guess and reset solver
  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

  # Solve again with new initial guess
  stats = solve!(solver, nlp, stats, print_level = 0)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end
