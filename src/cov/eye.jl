function coveye(x, z, hyp, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    assert(n==m)
    σ² = exp(2*hyp[1])    
    K = eye(n, m)*σ²
    return K
end

function partial_coveye(x, z, hyp, i, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    assert(n==m)
    σ² = exp(2*hyp[1])
    if i==1
        K = eye(n, m)*σ²*2
    else
        error("Unknown hyperparamter")
    end
    return K
end

tags_eye = ["noise"]

function covEye(hyp=[0.0])
    obj = CovarianceFunction(:covEye, 
                             coveye, 
                             partial_coveye, 
                             hyp)
    obj.spec["tag"] = tags_eye
    return obj                             
end

export covEye
