function covzero(x, z, hyp, fvec, spec)
    n = size(x, 1)
    m = size(m, 1)
    K = zeros(n, m)
    return K
end

function partial_covzero(x, z, hyp, i, fvec, spec)
    error("Unknown hyperparamter")
end

tags_zero = ["const"]

function covZero()
    obj = CovarianceFunction(:covZero, 
                             covzero, 
                             partial_covzero, 
                             [])
    obj.spec["tag"] = tags_zero
    return obj
end

covNone = covZero()

export covZero, covNone
