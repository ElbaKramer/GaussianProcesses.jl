function covtanh(x, z, hyp, fvec)
    n = length(fvec)
    assert(n==1)
    covfunc = fvec[1]
    K = covmat(covfunc, x, z, hyp)
    K = tanh(K)
    return K
end

function partial_covtanh(x, z, hyp, fvec, i)
    n = length(fvec)
    assert(n==1)
    covfunc = fvec[1]
    K = covmat(covfunc, x, z, hyp)
    K = (1 - tan(K).^2) * partial_covmat(covfunc, x, z, i, hyp)
    return K
end

function covTanh(hyp=[], fvec=[])
    return CompositeCovarianceFunction(:covTanh, 
                                       covtanh, 
                                       partial_covtanh, 
                                       hyp,
                                       fvec)
end

export covTanh
