function covposdef(x, z, hyp, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    tasks = spec["tasks"]
    irank = spec["irank"]
    ntasks = length(tasks)
    idx = indexin(x, tasks)
    idz = indexin(z, tasks)
    L = vec2low(hyp, ntasks, irank)
    K = L*L'
    K = K[idx,idz]
    return K
end

function partial_covposdef(x, z, hyp, i, fvec, spec)
    n = size(x, 1)
    m = size(z, 1)
    tasks = spec["tasks"]
    irank = spec["irank"]
    ntasks = length(tasks)
    nlf = convert(Integer, irank*(2*ntasks-irank+1)/2)
    idx = indexin(x, tasks)
    idz = indexin(z, tasks)
    if i<=nlf
        L = vec2low(hyp, ntasks, irank)
        o, p = ind2mind(i, ntasks, irank)
        J = zeros(ntasks, ntasks)
        J[o,p] = 1
        K = J*L' + L*J'
        K = K[idx,idz]
    else
        error("Unknown hyperparamter")
    end
    return K
end

function covPosDef(tasks; irank=length(tasks))
    ntasks = length(tasks)
    nlf = convert(Integer, irank*(2*ntasks-irank+1)/2)
    L = chol(eye(nlf, nlf))
    hyp = low2vec(L, ntasks, irank)
    obj = CovarianceFunction(:covPosDef, 
                             covposdef, 
                             partial_covposdef, 
                             hyp)
    obj.spec["tasks"] = tasks
    obj.spec["irank"] = irank
    return obj
end

function vec2low(v, D, irank)
    L = zeros(D,D)
    low = 1
    for i in 1:D
        up = min(irank, i)
        L[i,1:up] = v[low:low+up-1]
        low = low + up
    end
    return L
end

function low2vec(L, D, irank)
    v = []
    for i in 1:D
        v = apply(vcat, v, L[i,1:min(i,irank)])
    end
    return v
end

function ind2mind(pos, D, irank)
    i = 0
    j = 1
    s = 0
    sn = 0
    while sn < pos
        i = i + 1
        s = sn
        sn = s + min(irank, i)
    end
    if i > 0
        j = pos - s
    end
    return i, j
end

export covPosDef
