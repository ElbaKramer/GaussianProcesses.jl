function linear(x, z, hyp, fvec, spec)
    σ² = exp(2*hyp[1])
    shift = hyp[2]
    x = x.-shift
    z = z.-shift
    K = σ²*x*z'
    return K
end

function partial_linear(x, z, hyp, i, fvec, spec)
    σ² = exp(2*hyp[1])
    shift = hyp[2]
    x = x.-shift
    z = z.-shift
    if i==1
        K = 2*σ²*x*z'
    elseif i==2
        n = size(x, 1)
        m = size(z, 1)
        K = -σ² * (repeat(x, outer=(1, m)) + repeat(z', outer=(n, 1)))
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covLinear(hyp=[0.0,0.0])
    obj  = CovarianceFunction(:covLinear,
                              linear, 
                              partial_linear, 
                              hyp)
    return obj
end

export covLinear
