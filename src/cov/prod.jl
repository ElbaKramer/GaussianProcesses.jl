function covprod(x, z, hyp, fvec, spec)
    n = length(fvec)
    v = vcat([fill(i, numhyp(fvec[i])) for i in 1:n]...)
    K = 1
    for i=1:n
        K = K .* covmat(fvec[i], x, z, hyp[v.==i])
    end
    return K
end

function partial_covprod(x, z, hyp, i, fvec, spec)
    n = length(fvec)
    v = vcat([fill(j, numhyp(fvec[j])) for j in 1:n]...)
    if i <= length(v)
        K = 1
        vi = v[i]
        j = sum(v[1:i].==vi)
        for ii in 1:n
            if ii == vi
                K = K .* partial_covmat(fvec[ii], x, z, j, hyp[v.==ii])
            else
                K = K .* covmat(fvec[ii], x, z, hyp[v.==ii])
            end
        end
        return K
    else
        error("Unknown hyperparameter")
    end
end

tags_prod = ["product"]

function covProd(fvec=[])
    obj = CovarianceFunction(:covProd, 
                             covprod, 
                             partial_covprod, 
                             [])
    obj.fvec = fvec
    obj.tags = tags_prod
    return obj
end

import Base.*

function *(f::CovarianceFunction)
    return f
end

function *(f1::CovarianceFunction, f2::CovarianceFunction)
    fn1 = f1.fname == :covProd
    fn2 = f2.fname == :covProd
    if !fn1 && !fn2
        f = covProd()
        f.fvec = [deepcopy(f1), deepcopy(f2)]
        return f
    elseif fn1 && !fn2
        f = deepcopy(f1)
        append!(f.fvec, [deepcopy(f2)])
        return f
    elseif !fn1 && fn2
        f = deepcopy(f2)
        prepend!(f.fvec, [deepcopy(f1)])
        return f
    else
        f = deepcopy(f1)
        append!(f.fvec, deepcopy(f2.fvec))
        return f
    end
end

export covProd, *
