type CovarianceFunction
    fname::Symbol
    f::Function
    pf::Function
    hyp::Vector{Float64}
    fvec::Vector{CovarianceFunction}
    spec::Dict{String,Any}
    function CovarianceFunction(fname::Symbol, f::Function, pf::Function, hyp)
        return new(fname, f, pf, hyp, Array(CovarianceFunction,0), Dict{String,Any}())
    end
end

function covmat(f::CovarianceFunction,
                x::Array, z::Array,
                hyp::Vector=gethyp(f))
    K = f.f(x, z, hyp, f.fvec, f.spec)
    return K
end

function partial_covmat(f::CovarianceFunction,
                        x::Array, z::Array, i::Integer,
                        hyp::Vector=gethyp(f))
    pK = f.pf(x, z, hyp, i, f.fvec, f.spec)
    return pK
end

function numhyp(f::CovarianceFunction)
    return length(gethyp(f))
end

function numhyp(f::CovarianceFunction, self::Bool=false)
    nhyp = length(gethyp(f, true))
    if !self
        nf = length(f.fvec)
        for i in 1:nf
            nhyp = nhyp + numhyp(f.fvec[i])
        end
    end
    return nhyp
end

function gethyp(f::CovarianceFunction, self::Bool=false)
    if self
        return f.hyp
    else
        nf = length(f.fvec)
        hyps = [gethyp(f.fvec[i]) for i in 1:nf]
        hyp = apply(vcat, f.hyp, hyps)
        return hyp
    end
end

function sethyp!(f::CovarianceFunction, hyp::Vector, self::Bool=false)
    if length(hyp) != numhyp(f)
        error("Length does not match")
    elseif self
        f.hyp = hyp
    else
        if !isempty(f.hyp)
            snhyp = length(f.hyp)
            shyp = hyp[1:snhyp]
            hyp = hyp[(snhyp+1):end]
            f.hyp = shyp
        end
        nf = length(f.fvec)
        v = apply(vcat, [fill(i, numhyp(f.fvec[i])) for i in 1:nf])
        for i in 1:nf
            sethyp!(f.fvec[i], hyp[v.==i])
        end
    end
end

import Base.show

function show(io::IO, x::CovarianceFunction)
    print(io, x.fname, "(hyp=", string(x.hyp))
    print(io, ",fvec=[", join([string(f) for f in x.fvec], ","), "])")
end

function hastag(f::CovarianceFunction, tags::String...)
    ftag = get(f.spec, "tag", [])
    return all([t in ftag for t in tags])
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
    if hastag(f, "noise")
        return true
    elseif hastag(f, "product")
        anynoise = any([isnoise(ff) for ff in f.fvec])
        return anynoise
    else
        return false
    end
end

function remove_noise(f::CovarianceFunction)
    if hastag(f, "sum")
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
    elseif hastag(f, "noise")
        error("Trying to remove noise from noise")
    end
    return f
end

export CovarianceFunction, 
       SimpleCovarianceFunction, CompositeCovarianceFunction,
       covmat, partial_covmat,
       numhyp, gethyp, sethyp!,
       show, hastag,
       isnoise, remove_noise
