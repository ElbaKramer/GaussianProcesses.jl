function covlin(x, z, hyp, fvec, spec)
    σ² = exp(2*hyp[1])
    K = σ²*x*z'
    return K
end

function partial_covlin(x, z, hyp, i, fvec, spec)
    σ² = exp(2*hyp[1])
    if i==1
        K = 2*σ²*x*z'
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covLin(hyp=[0.0])
    obj  = CovarianceFunction(:covLin,
                              covlin, 
                              partial_covlin, 
                              hyp)
    return obj
end

export covLin
