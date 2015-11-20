function rqiso(x, z, hyp, fvec, spec)
    ℓ = exp(hyp[1])
    σ² = exp(2*hyp[2])
    α = exp(hyp[3])
    K = sqdist(x'/ℓ, z'/ℓ)
    K = σ²*(1+0.5*K/α).^(-α)
    return K
end

function partial_rqiso(x, z, hyp, i, fvec, spec)
    ℓ = exp(hyp[1])
    σ² = exp(2*hyp[2])
    α = exp(hyp[3])
    D2 = sqdist(x'/ℓ, z'/ℓ)
    if i==1
        K = σ²*(1+0.5*D2/α).^(-α).*D2
    elseif i==2
        K = 2*σ²*(1+0.5*D2/α).^(-α)
    elseif i==3
        K = (1+0.5*D2/α)
        K = σ²*K.^(-α).*(0.5*D2./K - α*log(K))
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covRQiso(hyp=[0.0,0.0,0.0])
    obj = CovarianceFunction(:covRQiso,
                             rqiso, 
                             partial_rqiso, 
                             hyp)
    return obj
end

export covRQiso
