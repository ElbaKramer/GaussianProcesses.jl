function sqexpiso(x, z, hyp, fvec, spec)
    ℓ = exp(hyp[1])
    σ² = exp(2*hyp[2])
    K = sqdist(x'/ℓ, z'/ℓ)
    K = σ²*exp(-K/2)
    return K
end

function partial_sqexpiso(x, z, hyp, i, fvec, spec)
    ℓ = exp(hyp[1])
    σ² = exp(2*hyp[2])
    K = sqdist(x'/ℓ, z'/ℓ)
    if i==1
        K = σ²*exp(-K/2).*K
    elseif i==2
        K = 2*σ²*exp(-K/2)
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covSEiso(hyp=[0.0,0.0])
    obj = CovarianceFunction(:covSEiso,
                             sqexpiso, 
                             partial_sqexpiso, 
                             hyp)
    return obj
end

export covSEiso
