function covtci(x, z, hyp, fvec)
    sfs = hyp[1]
    ellp = hyp[2]
    p = hyp[3]
    sfp = hyp[4]
    sfn = hyp[5]
    f = covSEiso(hyp=[p, sfs]) + covPeriodicNoDC(hyp=[ellp, p, sfp]) + covNoise(hyp=[sfn])
    K = covmat(f, x, z, hyp)
    return K
end

function partial_covtci(x, z, hyp, fvec, i)
    sfs = hyp[1]
    ellp = hyp[2]
    p = hyp[3]
    sfp = hyp[4]
    sfn = hyp[5]
    f = covSEiso(hyp=[p, sfs]) + covPeriodicNoDC(hyp=[ellp, p, sfp]) + covNoise(hyp=[sfn])
    if i == 1
        K = partial_covmat(f, x, z, 2)
    elseif i == 2
        K = partial_covmat(f, x, z, 3)
    elseif i == 3
        K = partial_covmat(f, x, z, 1) + partial_covmat(f, x, z, 4)
    elseif i == 4
        K = partial_covmat(f, x, z, 5)
    elseif i == 5
        K = partial_covmat(f, x, z, 6)
    else
        error("Unknown Hyperparameter")
    end
    return K
end

function covTci(hyp=[0.0,0.0,0.0,0.0,0.0])
    return SimpleCovarianceFunction(:covTci, 
                                    covtci, 
                                    partial_covtci, 
                                    hyp)
end

export covTci
