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

tags_mask = ["wrapper", "mask"]

function covMask(covfunc, mask)
    obj = CovarianceFunction(:covMask, 
                             covmask, 
                             partial_covmask, 
                             [])
    obj.fvec = [covfunc]
    obj.spec["mask"] = mask
    obj.tags = tags_mask
    return obj
end

export covMask
