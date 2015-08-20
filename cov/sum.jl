function covsum(x, z, hyp, fvec)
    n = length(fvec)
    v = apply(vcat, [fill(i, numhyp(fvec[i])) for i in 1:n])
    K = 0
    for i in 1:n
        K = K + covmat(fvec[i], x, z, hyp[v.==i])
    end
    return K
end

function partial_covsum(x, z, hyp, fvec, i)
    n = length(fvec)
    v = apply(vcat, [fill(j, numhyp(fvec[j])) for j in 1:n])
    if i <= length(v)
        vi = v[i]
        j = sum(v[1:i].==vi)
        K = partial_covmat(fvec[vi], x, z, j, hyp[v.==vi])
        return K
    else
        error("Unknown hyperparameter")
    end
end

function covSum(fvec=[])
    return CompositeCovarianceFunction(:covSum, 
                                       covsum, 
                                       partial_covsum, 
                                       [],
                                       fvec)
end

function +(f1::CovarianceFunction, f2::CovarianceFunction)
    fn1 = f1.fname == :covSum
    fn2 = f2.fname == :covSum
    if !fn1 && !fn2
        f = covSum()
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

export covSum, +
