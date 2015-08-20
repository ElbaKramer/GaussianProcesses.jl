function covmask(x, z, hyp, fvec)
    covfunc = fvec[1]
    mask = fvec[2]
    K = covmat(covfunc, x[:,mask], z[:,mask], hyp)
    return K
end

function partial_covmask(x, z, hyp, fvec, i)
    covfunc = fvec[1]
    mask = fvec[2]
    if i <= length(hyp)
        K = partial_covmat(covfunc, x[:,mask], z[:,mask], i, hyp)
    else
        error("Unknown hyperparameter")
    end
    return K
end

function covMask(covfunc, mask, inputdim)
    if !(typeof(mask) <: BitArray)
        mask = [i in mask for i in [1:inputdim]]
    end
    return CompositeCovarianceFunction(:covMask, 
                                       covmask, 
                                       partial_covmask, 
                                       [],
                                       [covfunc, mask])
end

export covMask
