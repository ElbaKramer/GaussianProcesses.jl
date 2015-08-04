import Optim.DifferentiableFunction
import Optim.optimize

# original train function using juila's Optim package
# but this didn't work well making errors, (no convergence)
function train_notused(gp::GaussianProcess, x, y, iter=100, verbose=true)
    # objective function
    function f(hyp)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        L = chol(Σ)
        α = solvechol(L, y-μ)
        nlml = dot(y-μ, α/2) + sum(log(diag(L))) + n*log(2π)/2
        return nlml
    end
    # first gradient function
    function g!(hyp, dnlml)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        L = chol(Σ)
        α = solvechol(L, y-μ)
        Q = solvechol(L, eye(n)) - α*α'
        for i in 1:length(dnlml)
            dnlml[i] = sum(sum(Q.*partial_covmat(gp.covfunc, x, x, i, hyp)))/2
        end
        dnlml = dnlml/norm(dnlml)
    end
    # evaluate both logliklihood and gradient
    function fg!(hyp, dnlml)
        n = size(x, 1)
        μ = meanvec(gp.meanfunc, x)
        Σ = covmat(gp.covfunc, x, x, hyp)
        L = chol(Σ)
        α = solvechol(L, y-μ)
        nlml = dot(y-μ, α/2) + sum(log(diag(L))) + n*log(2π)/2
        Q = solvechol(L, eye(n)) - α*α'
        for i in 1:length(dnlml)
            dnlml[i] = sum(Q.*partial_covmat(gp.covfunc, x, x, i, hyp))/2
        end
        dnlml = dnlml/norm(dnlml)
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
    return opt
end
