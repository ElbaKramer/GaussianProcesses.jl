function lik_all(hyp, gp::GaussianProcess, x, y)
    ndata = length(x)
    nlml = 0.0
    dnlml = zeros(length(hyp))
    for i in 1:ndata
        nlmli, dnlmli = lik(hyp, gp, x[i], y[i])
        nlml = nlml + nlmli
        dnlml = dnlml + dnlmli
    end
    return nlml, dnlml
end

function lik_all(gp::GaussianProcess, x, y)
    return lik_all(gethyp(gp.covfunc), x, y)
end

function train_all(gp::GaussianProcess, x, y, iter=500)
    hyp, evals, iters = minimize(gethyp(gp.covfunc), lik_all, iter, gp, x, y)
    return hyp
end

function train_all!(gp::GaussianProcess, x, y, iter=500)
    hyp = train_all(gp, x, y, iter)
    sethyp!(gp.covfunc, hyp)
    return hyp
end

export lik_all, train_all, train_all!
