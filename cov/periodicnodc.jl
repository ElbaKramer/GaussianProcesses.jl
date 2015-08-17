function embi0(x)
    y = 3.75/x
    y = 0.39894228 + 0.01328592*y + 0.00225319*y^2 - 0.00157565*y^3 +
        0.00916281*y^4 - 0.02057706*y^5 + 0.02635537*y^6 - 0.01647633*y^7 +
        0.00392377*y^8
    y = y/sqrt(x)
    return y
end

function embi1(x)
    y = 3.75/x
    y = 0.39894228     - 0.03988024*y   - 0.00362018*y^2 + 0.00163801*y^3 - 
        0.01031555*y^4 + 0.02282967*y^5 - 0.02895312*y^6 + 0.01787654*y^7 - 
        0.00420059*y^8
    y = y/sqrt(x)
    return y
end

function covD(D, ell)
    if ell>1e4
        K = cos(D)
    elseif 1/ell^2<3.75
        K = exp(cos(D)/ell^2)
        b0 = besseli(0,1/ell^2)
        K = (K-b0)/(exp(1/ell^2)-b0)
    else
        K = exp((cos(D)-1)/ell^2)
        b0 = embi0(1/ell^2)
        K = (K-b0)/(1-b0)
    end
    return K
end

function periodicnodc(x, z, hyp)
    ℓ = exp(hyp[1])
    p = exp(hyp[2])
    σ² = exp(2*hyp[3])
    K = sqrt(sqdist(x', z'))
    K = 2*π*K/p
    K = σ²*covD(K, ℓ)
    return K
end

function partial_periodicnodc(x, z, hyp, i)
    ell = exp(hyp[1])
    p = exp(hyp[2])
    sf2 = exp(2*hyp[3])
    K = sqrt(sqdist(x', z'))
    K = 2*pi*K/p
    if i==1
        if ell>1e4                                 
            K = zeros(size(K))
        elseif 1/ell^2<3.75
            cK = cos(K); ecK = exp(cK/ell^2)
            b0 = besseli(0,1/ell^2)
            b1 = besseli(1,1/ell^2)
            K = 2*(exp(1/ell^2)-ecK)*b1 -
                2*(exp(1/ell^2)-ecK.*cK)*b0 + 
                4*exp(2*(cos(K/2)/ell).^2).*sin(K/2).^2
            K = sf2/(ell*(exp(1/ell^2)-b0))^2 * K
        else
            cK = cos(K); ecK = exp((cK-1)/ell^2)
            b0 = embi0(1/ell^2)
            b1 = embi1(1/ell^2)
            K = 2*(1-ecK)*b1 - 2*(1-ecK.*cK)*b0 + 
                4*exp(2*(cos(K/2).^2-1)/ell^2).*sin(K/2).^2
            K = sf2/(ell*(1-b0))^2 * K
        end
    elseif i==2
        if ell>1e4                                  
            K = sf2*sin(K).*K
        elseif 1/ell^2<3.75
            K = exp(cos(K)/ell^2).*sin(K).*K
            K = sf2/ell^2/(exp(1/ell^2)-besseli(0,1/ell^2))*K
        else
            K = exp((cos(K)-1)/ell^2).*sin(K).*K
            K = sf2/ell^2/(1-embi0(1/ell^2))*K
        end
    elseif i==3
        K = 2*sf2*covD(K,ell)
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covPeriodicNoDC(hyp=[0.0,0.0,0.0])
    return SimpleCovarianceFunction(:covPeriodicNoDC,
                                    periodicnodc, 
                                    partial_periodicnodc, 
                                    hyp)
end

export covPeriodicNoDC
