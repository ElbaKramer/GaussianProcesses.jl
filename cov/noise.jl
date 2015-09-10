function noise(x, z, hyp, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    σ² = exp(2*hyp[1])
    tol = eps()
    T = sqdist(x', z').<tol*tol
    K = zeros(n, m)
    K[find(T)] = σ²
    return K
end

function partial_noise(x, z, hyp, i, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    σ² = exp(2*hyp[1])
    tol = eps()
    T = sqdist(x', z').<tol*tol
    K = zeros(n, m)
    if i==1
        K[find(T)] = 2*σ²
    else
        error("Unknown hyperparamter")
    end
    return K
end

tags_noise = ["noise"]

function covNoise(hyp=[0.0])
    obj = CovarianceFunction(:covNoise, 
                             noise, 
                             partial_noise, 
                             hyp)
    obj.spec["tag"] = tags_noise
    return obj
end

export covNoise
