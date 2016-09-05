using DataFrames
using GLM

data = readtable("data/air.csv")
OLS = glm(count~time, data, Normal(), IdentityLink())

println(OLS)

