type GaussianProcess
    meanfunc::MeanFunction
    covfunc::CovarianceFunction
end

function lik(gp::GaussianProcess, x, y)
    # size of data
    n = size(x, 1)
    # mean vector and covariance matrix
    μ = meanvec(gp.meanfunc, x)
    Σ = covmat(gp.covfunc, x, x)

    # calculate inverse of Σ using cholesky factorization
    # here, α = Σ⁻¹(y-μ)
    L = chol(Σ)
    α = solvechol(L, y-μ)
    # calculate negative log marginal likelihood
    # here, nlml = 1/2*(y-μ)ᵀΣ⁻¹(y-μ) + 1/2*log(|Σ|) + k/2*log(2π)
    nlml = dot(y-μ, α/2) + sum(log(diag(L))) + n*log(2π)/2

    # precompute Q
    Q = solvechol(L, eye(n)) - α*α'
    # preallocate memory
    dnlml = zeros(numhyp(gp.covfunc))
    # calculate partial derivatives
    for i in 1:length(dnlml)
        dnlml[i] = sum(sum(Q.*partial_covmat(gp.covfunc, x, x, i)))/2
    end

    # return negative log marginal likelihodd and its gradient
    return nlml, dnlml
end

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

# this new train function uses gpml matlab codes with MATALB julia module
include("gpml.jl")
function train(gp::GaussianProcess, x, y, iter=1000, verbose=true)
    hyp = minimize(gp, x, y, iter, verbose)
    return hyp
end

function train!(gp::GaussianProcess, x, y, iter=1000, verbose=true)
    # just get optimum value calling train funciton
    hyp = train(gp, x, y, iter, verbose)
    # set the hyperparameters with the new one
    sethyp!(gp.covfunc, hyp)

    # return the result also
    return hyp
end

function predict(gp::GaussianProcess, x, y, xs)
    # original covfunc and its noise free version
    covgiven = gp.covfunc
    covsignal = remove_noise(gp.covfunc)
    println(covgiven)

    # mean vectors
    μs = meanvec(gp.meanfunc, xs)
    μx = meanvec(gp.meanfunc, x)
    # covariance matrices
    Kss = covmat(covsignal, xs, xs)
    Ksx = covmat(covsignal, xs, x)
    Kxx = covmat(covgiven, x, x)
    Lxx = chol(Kxx)

    # compute conditional
    μ = μs + Ksx*solvechol(Lxx, y-μx)
    Σ = Kss - Ksx*solvechol(Lxx, Ksx')
    σ²= diag(Σ)

    # return mean and variance with length of test input
    return μ, σ²
end

function test(gp::GaussianProcess, x, y, xs, ys)
    # use results from predict
    μ, σ² = predict(gp, x, y, xs)

    # pointwise log probabilities
    lp = -(y-μ).^2./σ²/2-log(2π*n2)/2

    # return mean, variance and log probability with length of test input
    return μ, σ², lp
end

export GaussianProcess,
       lik, train, train!, predict, test
