import Optim.DifferentiableFunction
import Optim.optimize

function train_optim(gp::GaussianProcess, x, y, iter)
    verbose = true
    # objective function
    function f(hyp)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        if !isposdef(Σ)
            return NaN
        end
        L = chol(Σ)
        α = solvechol(L, y-μ)
        nlml = dot(y-μ, α)/2 + sum(log(diag(L))) + n*log(2π)/2
        return nlml
    end
    # first gradient function
    function g!(hyp, dnlml)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        if !isposdef(Σ)
            dnlml = fill(NaN, size(dnlml))
            return dnlml
        end
        L = chol(Σ)
        α = solvechol(L, y-μ)
        Q = solvechol(L, eye(n)) - α*α'
        for i in 1:length(dnlml)
            dnlml[i] = sum(sum(Q.*partial_covmat(gp.covfunc, x, x, i, hyp)))/2
        end
        return dnlml
    end
    # evaluate both logliklihood and gradient
    function fg!(hyp, dnlml)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        if !isposdef(Σ)
            dnlml = fill(NaN, size(dnlml))
            return NaN
        end
        L = chol(Σ)
        α = solvechol(L, y-μ)
        Q = solvechol(L, eye(n)) - α*α'
        nlml = dot(y-μ, α)/2 + sum(log(diag(L))) + n*log(2π)/2        
        for i in 1:length(dnlml)
            dnlml[i] = sum(sum(Q.*partial_covmat(gp.covfunc, x, x, i, hyp)))/2
        end
        return nlml
    end

    # define defferentiable function for optim package
    d4 = DifferentiableFunction(f, g!, fg!)
    # optimize hyperparameters
    hyp = gethyp(gp.covfunc)
    opt = optimize(d4, hyp,
                   method = :l_bfgs,
                   show_trace = verbose)

    # return the result of optimize
    return opt.minimum
end

export train_optim
