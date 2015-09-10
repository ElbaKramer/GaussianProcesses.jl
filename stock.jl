println("Loading packages")
using GaussianProcesses
using PyPlot

println("Reading data")
data, header = readcsv("data/stock.csv", header=true)

s = data[:,1]
x = data[:,2]
y = data[:,3]

s = indexin(s, unique(s))
x = hcat(s,x)
x = convert(Array{Float64, 2}, x)
y = convert(Array{Float64, 1}, y)

learning = false
if learning

println("Generating gp object")
meanfunc = meanZero()
covfunc = covPosDef(unique(x[:,1])) + covEye()
gp1 = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp1)

println("Optimizing hyperparameters of covariance kernel")
println("initial params = ", gethyp(gp1.covfunc))
opt1 = train!(gp1, x[:,1], y, -300)
println("optimized params = ", gethyp(gp1.covfunc))

println("Generating gp object")
meanfunc = meanZero()
covfunc = covSEiso() + covEye()
gp2 = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp2)

println("Optimizing hyperparameters of covariance kernel")
println("initial params = ", gethyp(gp2.covfunc))
opt2 = train_all!(gp2, {x[s.==1,2], x[s.==2,2], x[s.==3,2]}, {y[s.==1], y[s.==3], y[s.==3]}, -500)
sethyp!(gp2.covfunc, opt2)
println("optimized params = ", gethyp(gp2.covfunc))

println("Generating gp object")
meanfunc = meanZero()
covfunc = covMask(covPosDef(unique(x[:,1])), 1, 2) * covMask(covSEiso() + covEye(), 2, 2)
sethyp!(covfunc, vcat(opt1[1:end-1], opt2[1:end-1], mean([opt1[end], opt2[end]])))
gp = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp)

println("Optimizing hyperparameters of covariance kernel")
println("initial params = ", gethyp(gp.covfunc))
opt = train!(gp, x, y, -500)
println("optimized params = ", gethyp(gp.covfunc))

end

println("Generating gp object")
meanfunc = meanZero()
covfunc = covMask(covPosDef(unique(x[:,1])), 1, 2) * covMask(covSEiso() + covEye(), 2, 2)
final_opt = [37.224864552426745, 31.26115653843799, 5.104548314604684, 39.16322106016218, 3.944320721108208, 3.985199913012164, -3.3204166283944416, -0.8044792682823634, -4.269709734098495]
sethyp!(covfunc, final_opt)
gp = GaussianProcess(meanfunc, covfunc)
println("gp = ", gp)

symbols = {"GE", "MSFT", "XOM"}

for i in 1:3
    figure()
    hold(true)
    for j in 1:3
        plot(x[s.==j,2], y[s.==j])
    end
    m, s2, lp = test(gp, x[s.!=i,:], y[s.!=i], x[s.==i,:], y[s.==i])
    plot(x[s.==i,2], m)
    hold(false)
    title(string("Predicting ", symbols[i]))
    figure()
    plot(lp)
    title(string("Log prob of ", symbols[i]))
end

show()
exit()
