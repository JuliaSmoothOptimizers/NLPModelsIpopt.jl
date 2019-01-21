using NLPModelsIpopt, NLPModels, Ipopt, Test

function tests()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  x, f, c, λ, zL, zU, status = ipopt(nlp)
  @test isapprox(x, [1.0; 1.0], rtol=1e-6)

  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   c=x->[sum(x) - 1.0], lcon=[0.0], ucon=[0.0])
  x, f, c, λ, zL, zU, status = ipopt(nlp)
  @test isapprox(x, [-1.4; 2.4], rtol=1e-6)
end

tests()
