println("Loading packages")
using GaussianProcesses
using PyPlot

println("Reading data")
data, header = readcsv("data/air.csv", header=true)
x = data[:,1]
y = data[:,2]

println("Generating gp object")
meanfunc = meanZero()
covfunc = covPeriodicNoDC() + covNoise()
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
println("optimized nlml = ", lik(gp, x, y)[1])
m, s2 = predict(gp, x, y, x)

println("Plotting data and result")
figure()
plot(x, y)
hold(true)
plot(x, m)
hold(false)
print("Press any key to continue")
readline()

# println("Sample data given Gaussian Process prior")
# ys = sample(gp, x)
# figure()
# plot(x, ys)
# print("Press any key to continue")
# readline()
