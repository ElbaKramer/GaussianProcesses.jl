function covmask(x, z, hyp, fvec, spec)
    covfunc = fvec[1]
    mask = spec["mask"]
    K = covmat(covfunc, x[:,mask], z[:,mask], hyp)
    return K
end

function partial_covmask(x, z, hyp, i, fvec, spec)
    covfunc = fvec[1]
    mask = spec["mask"]
    if i <= numhyp(covfunc)
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
    obj = CovarianceFunction(:covMask, 
                             covmask, 
                             partial_covmask, 
                             [])
    obj.fvec = [covfunc]
    obj.spec["mask"] = mask
    return obj
end

export covMask
