function covtanh(x, z, hyp, fvec)
    covfunc = fvec[1]
    K = covmat(covfunc, x, z, hyp)
    K = tanh(K)
    return K
end

function partial_covtanh(x, z, hyp, fvec, i)
    covfunc = fvec[1]
    K = covmat(covfunc, x, z, hyp)
    if i <= length(hyp)
        K = (1 - tan(K).^2) * partial_covmat(covfunc, x, z, i, hyp)
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covTanh(covfunc)
    return CompositeCovarianceFunction(:covTanh, 
                                       covtanh, 
                                       partial_covtanh, 
                                       [],
                                       [covfunc])
end

export covTanh
