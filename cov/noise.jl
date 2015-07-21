function noise(x, z, hyp)
    σ² = exp(2*hyp[1])
    tol = eps()
    K = convert(Array{Float64,2}, sqdist(x', z').<tol*tol)

    K = σ²*K
    return K
end

function partial_noise(x, z, hyp, i)
    σ² = exp(2*hyp[1])
    tol = eps()
    K = convert(Array{Float64,2}, sqdist(x', z').<tol*tol)

    if i==1
        K = 2*σ²*K
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
