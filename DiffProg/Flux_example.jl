using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test #using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [2.2, 1.0, 2.0, 0.4]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)



function predict_rd()
  Array(solve(prob,Tsit5(),saveat=0.1,reltol=1e-4))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())

opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
Flux.train!(loss_rd, Flux.params(p), Iterators.repeated((), 100), opt, cb = cb)

