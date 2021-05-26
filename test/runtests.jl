using ADNLPModels, NLPModelsIpopt, NLPModels, Ipopt, Test

function tests()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
  @test stats.status == :first_order

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol=1e-12)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
  @test stats.status == :first_order

  # solve again from solution
  x0 = copy(stats.solution)
  stats = ipopt(nlp, x0=x0, tol=1e-12)
  @test isapprox(stats.solution, x0, rtol=1e-6)
  @test stats.status == :first_order
  @test stats.iter == 0

  function callback(alg_mod, iter_count, args...)
    return iter_count < 1
  end
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol=1e-12, callback=callback)
  @test stats.status == :user
  @test stats.solver_specific[:internal_msg] == :User_Requested_Stop
  @test stats.iter == 1

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   x->[sum(x) - 1.0], [0.0], [0.0])
  stats = ipopt(nlp)
  @test isapprox(stats.solution, [-1.4; 2.4], rtol=1e-6)
  @test stats.iter == 1
  @test stats.status == :first_order

  # solve constrained problem again from solution
  x0 = copy(stats.solution)
  y0 = copy(stats.multipliers)
  zL = copy(stats.multipliers_L)
  zU = copy(stats.multipliers_U)
  stats = ipopt(nlp, x0=x0, y0=y0, zL0=zL, zU0=zU)
  @test isapprox(stats.solution, x0, rtol=1e-6)
  @test isapprox(stats.multipliers, y0, rtol=1e-6)
  @test isapprox(stats.multipliers_L, zL, rtol=1e-6)
  @test isapprox(stats.multipliers_U, zU, rtol=1e-6)
  @test stats.iter == 0
  @test stats.status == :first_order

  meta = NLPModelMeta(1, x0 = rand(1), lvar = zeros(1), uvar = ones(1), minimize = false)
  nlp = ADNLPModel(meta, Counters(), ADNLPModels.ForwardDiffAD(), x -> x[1], x -> [])
  stats = ipopt(nlp)
  @test isapprox(stats.solution, ones(1), rtol=1e-6)
  @test isapprox(stats.objective, 1.0, rtol=1e-6)
  @test isapprox(stats.multipliers_L, zeros(1), atol=1e-6)
  @test isapprox(stats.multipliers_U, -ones(1), rtol=1e-6)
  @test stats.status == :first_order
end

tests()
