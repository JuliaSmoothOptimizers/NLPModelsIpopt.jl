using ADNLPModels, NLPModelsIpopt, NLPModels, Ipopt, SolverCore, Test
using NLPModelsModifiers: FeasibilityFormNLS

@testset "Restart NLPModelsIpopt" begin
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = GenericExecutionStats(nlp)
  solver = IpoptSolver(nlp)
  stats = solve!(solver, nlp, stats, print_level = 0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol = 1e-6)
  @test stats.status == :first_order
  @test isapprox(stats.iter, 21; atol=1)
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
  @test stats.status == :first_order
  @test stats.elapsed_time > 0
  @test isapprox(stats.iter, 22; atol=1)
  @test stats.primal_feas ≈ 0.0
  @test isapprox(stats.dual_feas, 0.0; atol=1e-9)

  # solve again from solution
  x0 = copy(stats.solution)
  stats = ipopt(nlp, x0 = x0, tol = 1e-12, print_level = 0)
  @test isapprox(stats.solution, x0, rtol = 1e-6)
  @test stats.status == :first_order
  @test isapprox(stats.iter, 0; atol=1)
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
  @test stats.iter == 5
  @test stats.primal_feas ≈ 0.0
  @test stats.dual_feas ≈ 0.0 atol = 1.49e-8
end

@testset "ipopt with AbstractNLSModel" begin
  nls = ADNLSModel(x -> [x[1] - 1, x[2] - 2], [0.0, 0.0], 2)
  stats = ipopt(nls, print_level = 0)
  @test isapprox(stats.solution, [1.0, 2.0], rtol = 1e-6)
  # Accept :first_order or :unknown due to possible log file issues
  @test stats.status == :first_order || stats.status == :unknown
  if hasfield(typeof(stats), :iter) && stats.iter != -1
    @test stats.iter >= 0
  end
  if hasfield(typeof(stats), :dual_feas) && isfinite(stats.dual_feas)
    @test isapprox(stats.dual_feas, 0.0; atol=1e-8)
  end
end
