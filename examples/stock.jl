println("Loading packages")
using GaussianProcesses
#using PyPlot

println("Reading data")
data, header = readcsv("..data/stock.csv", header=true)

s = data[:,1]
x = data[:,2]
y = data[:,3]

s = indexin(s, unique(s))
x = hcat(s,x)
rows = s .<= 2

x = convert(Array{Float64, 2}, x)
y = convert(Array{Float64, 1}, y)

println("Generating gp object")
meanfunc = meanZero()
covfunc = covMask(covPosDef(unique(x[rows,1])), 1, 2) * covMask(covSEiso()+covEye(), 2, 2)
gp = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp)

println("Evaluating negative log marginal likelihood of the model")
nlml, dnlml = lik(gp, x[rows,:], y[rows])
println("nlml = ", nlml)
println("dnlml = ", dnlml)

println("Individual sum")
row1 = s .== 1
row2 = s .== 2
nlml1, dnlml1 = lik(GaussianProcess(meanfunc, covSEiso()+covEye()), x[row1,2], y[row1])
nlml2, dnlml2 = lik(GaussianProcess(meanfunc, covSEiso()+covEye()), x[row2,2], y[row2])
println("nlml = ", nlml1+nlml2)
println("dnlml = ", dnlml1+dnlml2)

exit()
