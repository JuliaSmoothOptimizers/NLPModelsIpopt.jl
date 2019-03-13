using NLPModelsIpopt, NLPModels, Ipopt, Test

function tests()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   c=x->[sum(x) - 1.0], lcon=[0.0], ucon=[0.0])
  stats = ipopt(nlp)
  @test isapprox(stats.solution, [-1.4; 2.4], rtol=1e-6)

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol=1e-12)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)

  function callback(alg_mod, iter_count, args...)
    return iter_count < 1
  end
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = ipopt(nlp, tol=1e-12, callback=callback)
  @test stats.solver_specific[:internal_msg] == :User_Requested_Stop
end

tests()
