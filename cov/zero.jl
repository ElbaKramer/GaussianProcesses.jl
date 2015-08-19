function covzero(x, z, hyp)
    n = size(x, 1)
    m = size(m, 1)
    K = zeros(n, m)
    return K
end

function partial_covzero(x, z, hyp, i)
    error("Unknown hyperparamter")
end

function covZero(hyp=[])
    return SimpleCovarianceFunction(:covZero, 
                                    covzero, 
                                    partial_covzero, 
                                    hyp)
end

covNone = covZero()

export covZero, covNone
