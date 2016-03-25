println("Loading packages")
using GaussianProcesses
using PyPlot

println("Reading data")
data, header = readcsv("demo/data/air.csv", header=true)
x = data[:,1]
y = data[:,2]

figure()
plot(x,y)
title("Data shape")

println("Generating gp object")
meanfunc = meanZero()
covfunc = covSEiso() + covNoise()
gp = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp)

println("Evaluating negative log marginal likelihood of the model")
nlml, dnlml = lik(gp, x, y)
println("nlml = ", nlml)
println("dnlml = ", dnlml)

println("Optimizing hyperparameters of covariance kernel")
println("initial params = ", gethyp(gp.covfunc))
opt = train!(gp, x, y)
println("optimized params = ", gethyp(gp.covfunc))
nlml, dnlml = lik(gp, x, y)
println("optimized nlml = ", nlml)
m, s2, lp = test(gp, x, y, x, y)

println("Plotting data and result")
figure()
plot(x, y)
hold(true)
xs = collect(minimum(x):(1/12):(minimum(x)+1.2*(maximum(x)-minimum(x))))
m, s2 = predict(gp, x, y, xs)
plot(xs, m)
hold(false)
title("Plotting data and result")

println("Sample data given Gaussian Process prior")
ys = sample(gp, x)
figure()
plot(x, ys)
title("Sample data given Gaussian Process prior")
show()
