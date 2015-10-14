type CovarianceFunction
    # class specific
    fname::Symbol
    f::Function
    pf::Function
    tags::Vector{AbstractString}

    # object specific
    hyp::Vector{Float64}
    fvec::Vector{CovarianceFunction}

    # additional specs (either)
    spec::Dict{AbstractString,Any}

    # constructor
    function CovarianceFunction(fname::Symbol, f::Function, pf::Function, hyp)
        return new(fname, f, pf, Array(AbstractString,0), hyp, Array(CovarianceFunction,0), Dict{AbstractString,Any}())
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

function numhyp(f::CovarianceFunction, self::Bool=false)
    nhyp = length(gethyp(f, true))
    if !self && !isempty(f.fvec)
        nhyp = nhyp + sum([numhyp(ff) for ff in f.fvec])
    end
    return nhyp
end

function gethyp(f::CovarianceFunction, self::Bool=false)
    if self || isempty(f.fvec)
        return f.hyp
    else
        hyps = [gethyp(ff) for ff in f.fvec]
        hyp = vcat(f.hyp, hyps...)
        return hyp
    end
end

function sethyp!(f::CovarianceFunction, hyp::Vector, self::Bool=false)
    if length(hyp) != numhyp(f, self)
        error("Length does not match")
    elseif self || isempty(f.fvec)
        f.hyp = hyp
    else
        if !isempty(f.hyp)
            snhyp = length(f.hyp)
            shyp = hyp[1:snhyp]
            hyp = hyp[(snhyp+1):end]
            f.hyp = shyp
        end
        nf = length(f.fvec)
        v = vcat([fill(i, numhyp(f.fvec[i])) for i in 1:nf]...)
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

function hastag(f::CovarianceFunction, tags::AbstractString...)
    return issubset(tags, f.tags)
end

covdirname = "cov"
covdir = joinpath(dirname(@__FILE__), covdirname)
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
    elseif hastag(f, "product") || hastag(f, "wrapper")
        anynoise = any([isnoise(ff) for ff in f.fvec])
        return anynoise
    else
        return false
    end
end

using Iterators

function normal_form(f::CovarianceFunction)
    f = deepcopy(f)
    if hastag(f, "product")
        flen = length(f.fvec)
        fvec = [normal_form(ff) for ff in f.fvec]
        fvec = [begin
            if hastag(ff, "sum")
                ff.fvec
            else
                [ff]
            end
        end for ff in fvec]
        fvec = [*(p...) for p in product(fvec...)]
        f = +(fvec...)
    elseif hastag(f, "sum")
        flen = length(f.fvec)
        fvec = [normal_form(ff) for ff in f.fvec]
        f = +(fvec...)
    elseif hastag(f, "wrapper")
        fvec = f.fvec[1]
        if hastag(fvec, "sum") || hastag(fvec, "product")
            #TODO I think this can be improved
            ff = deepcopy(fvec)
            for i in 1:length(ff.fvec)
                fff = deepcopy(f)
                fff.fvec[1] = ff.fvec[i]
                ff.fvec[i] = fff
            end
            f = ff
        end
    end
    return f
end

function remove_noise(f::CovarianceFunction)
    f = normal_form(f)
    if hastag(f, "sum")
        keep = map(Bool,[!isnoise(ff) for ff in f.fvec]) # don't like this
        f.fvec = f.fvec[keep]
        f.fvec = [remove_noise(ff) for ff in f.fvec]
        flen = length(f.fvec)
        if flen == 1
            f = f.fvec[1]
        elseif flen == 0
            error("Removing noise results nothing left")
        end
    elseif hastag(f, "wrapper")
        f.fvec = remove_noise(f.fvec)
    elseif hastag(f, "noise")
        error("Trying to remove noise from only noise")
    end
    return f
end

export CovarianceFunction, 
       covmat, partial_covmat,
       numhyp, gethyp, sethyp!,
       show, hastag,
       isnoise, normal_form, remove_noise
