abstract CovarianceFunction

type SimpleCovarianceFunction <: CovarianceFunction
    fname::Symbol
    f::Function
    pf::Function
    hyp::Vector{Float64}
end

type CompositeCovarianceFunction <: CovarianceFunction
    fname::Symbol
    f::Function
    pf::Function
    hyp::Vector{Float64}
    fvec::Vector{Any}
end

function covmat(f::SimpleCovarianceFunction,
                x::Array, z::Array,
                hyp::Vector=gethyp(f))
    K = f.f(x, z, hyp)
    return K
end

function covmat(f::CompositeCovarianceFunction,
                x::Array, z::Array,
                hyp::Vector=gethyp(f))
    if isempty(f.fvec)
        error("We require at least one element")
    else
        K = f.f(x, z, hyp, f.fvec)
        return K
    end
end

function partial_covmat(f::SimpleCovarianceFunction,
                        x::Array, z::Array, i::Integer,
                        hyp::Vector=gethyp(f))
    pK = f.pf(x, z, hyp, i)
    return pK
end

function partial_covmat(f::CompositeCovarianceFunction,
                        x::Array, z::Array, i::Integer,
                        hyp::Vector=gethyp(f))
    if isempty(f.fvec)
        error("We require at least one element")
    else
        pK = f.pf(x, z, hyp, f.fvec, i)
        return pK
    end
end

function numhyp(f::SimpleCovarianceFunction)
    return length(gethyp(f))
end

function numhyp(f::CompositeCovarianceFunction, self::Bool=false)
    n = length(gethyp(f, true))
    if !self
        for i=1:length(f.fvec)
            n = n + numhyp(f.fvec[i])
        end
    end
    return n
end

function gethyp(f::SimpleCovarianceFunction)
    return f.hyp
end

function gethyp(f::CompositeCovarianceFunction, self::Bool=false)
    if self
        return f.hyp
    else
        n = length(f.fvec)
        hyps = [gethyp(f.fvec[i]) for i in 1:n]
        hyp = apply(vcat, f.hyp, hyps)
        return hyp
    end
end

function sethyp!(f::SimpleCovarianceFunction, hyp::Vector)
    if length(hyp) != numhyp(f)
        error("Length does not match")
    else
        f.hyp = hyp
    end
end

function sethyp!(f::CompositeCovarianceFunction, hyp::Vector)
    if length(hyp) != numhyp(f)
        error("Length does not match")
    else
        if !isempty(f.hyp)
            sn = length(f.hyp)
            shyp = hyp[1:sn]
            hyp = hyp[(sn+1):end]
            f.hyp = shyp
        end
        n = length(f.fvec)
        v = apply(vcat, [fill(i, numhyp(f.fvec[i])) for i in 1:n])
        for i=1:n
            sethyp!(f.fvec[i], hyp[v.==i])
        end
    end
end

import Base.show

function show(io::IO, x::SimpleCovarianceFunction)
    print(io, x.fname, "(hyp=", string(x.hyp), ")")
end

function show(io::IO, x::CompositeCovarianceFunction)
    print(io, x.fname, "(hyp=", string(x.hyp))
    print(io, ",fvec=[", join([string(f) for f in x.fvec], ","), "])")
end

covdir = "cov"
if isdir(covdir)
    for file in readdir(covdir)
        if splitext(file)[2] == ".jl"
            include(joinpath(covdir, file))
        end
    end
end

function isnoise(f::CovarianceFunction)
    if f.fname == :covNoise
        return true
    elseif f.fname == :covProd
        anynoise = any([isnoise(ff) for ff in f.fvec])
        return anynoise
    else
        return false
    end
end

function remove_noise(f::CovarianceFunction)
    if f.fname == :covSum
        f = deepcopy(f)
        flen = length(f.fvec)
        keep = [!isnoise(ff) for ff in f.fvec]
        f.fvec = f.fvec[keep]
        f.fvec = [remove_noise(ff) for ff in f.fvec]
        flen = length(f.fvec)
        if flen == 1
            f = f.fvec[1]
        elseif flen == 0
            error("This shouldn't happen")
        end
    elseif f.fname == :covNoise
        error("Trying to remove noise from noise")
    end
    return f
end

export CovarianceFunction, 
       SimpleCovarianceFunction, CompositeCovarianceFunction,
       covmat, partial_covmat,
       numhyp, gethyp, sethyp!,
       show,
       isnoise, remove_noise
