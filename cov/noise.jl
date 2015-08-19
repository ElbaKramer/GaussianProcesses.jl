function noise(x, z, hyp)
    n = size(x, 1)
    m = size(z, 1)
    σ² = exp(2*hyp[1])
    tol = eps()
    T = sqdist(x', z').<tol*tol
    K = zeros(n, m)
    K[find(T)] = σ²
    return K
end

function partial_noise(x, z, hyp, i)
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

function covNoise(hyp=[0.0])
    return SimpleCovarianceFunction(:covNoise, 
                                    noise, 
                                    partial_noise, 
                                    hyp)
end

export covNoise
