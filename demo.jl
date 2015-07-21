println("Loading packages")
using GaussianProcesses
using PyPlot

println("Reading data")
data, header = readcsv("data/air.csv", header=true)
x = data[:,1]
y = data[:,2]

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
println("Initial params: ", gethyp(gp.covfunc))
opt = train!(gp, x, y)
println("Optimized params: ", gethyp(gp.covfunc))
m, s2 = predict(gp, x, y, x)

println("Plotting data and result")
figure()
plot(x, y)
hold(true)
plot(x, m)
print("Press any key to continue")
readline()
