rNorm = rNorm0 = norm(Fx)
mul!(Jtu, Jx', Fx)
ArNorm = ArNorm0 = norm(Jtu)

solver.stats.rNorm0 = rNorm0
solver.stats.ArNorm0 = ArNorm0