struct MeanFunction
    fname::Symbol
    f::Function
    pf::Function
    hyp::Vector{Float64}
end

function meanvec(f::MeanFunction,
                 x::Array,
                 hyp::Vector=gethyp(f))
    μ = f.f(x, hyp)
    return μ
end

function partial_meanvec(f::MeanFunction,
                         x::Array,
                         hyp::Vector=gethyp(f))
    pμ = f.pf(x, hyp, i)
    return pμ
end

function gethyp(f::MeanFunction)
    return f.hyp
end

meandirname = "mean"
meandir = joinpath(dirname(@__FILE__), meandirname)
if isdir(meandir)
    for file in readdir(meandir)
        if splitext(file)[2] == ".jl"
            include(joinpath(meandir, file))
        end
    end
end

import Base.show

function show(io::IO, x::MeanFunction)
    print(io, x.fname, "(hyp=", string(x.hyp), ")")
end

export MeanFunction,
       meanvec, partial_meanvec,
       show
