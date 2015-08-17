function covconst(x, z, hyp)
    σ² = exp(2*hyp[1])
    n = size(x, 1)
    m = size(z, 1)
    K = σ²*ones(n, m)
    return K
end

function partial_covconst(x, z, hyp, i)
    σ² = exp(2*hyp[1])
    n = size(x, 1)
    m = size(z, 1)
    K = σ²*ones(n, m)
    if i==1
        K = 2*K
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covConst(hyp=[0.0])
    return SimpleCovarianceFunction(:covConst,
                                    covconst, 
                                    partial_covconst, 
                                    hyp)
end

export covConst
