function covtci(x, z, hyp, fvec, spec)
    sfs = hyp[1]
    ellp = hyp[2]
    p = hyp[3]
    sfp = hyp[4]
    sfn = hyp[5]
    newhyp = [p, sfs, ellp, p, sfp, sfn]
    f = covSEiso(newhyp[[1:2]]) + covPeriodicNoDC(newhyp[[3:5]]) + covNoise(newhyp[[6]])
    K = covmat(f, x, z)
    return K
end

function partial_covtci(x, z, hyp, i, fvec, spec)
    sfs = hyp[1]
    ellp = hyp[2]
    p = hyp[3]
    sfp = hyp[4]
    sfn = hyp[5]
    newhyp = [p, sfs, ellp, p, sfp, sfn]
    f = covSEiso(newhyp[[1:2]]) + covPeriodicNoDC(newhyp[[3:5]]) + covNoise(newhyp[[6]])
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
    obj = CovarianceFunction(:covTci, 
                             covtci, 
                             partial_covtci,
                             hyp)
    return obj
end

export covTci
