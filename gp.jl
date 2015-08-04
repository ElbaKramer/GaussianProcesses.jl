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

function lik(hyp, gp::GaussianProcess, x, y)
    # size of data
    n = size(x, 1)
    # mean vector and covariance matrix
    μ = meanvec(gp.meanfunc, x)
    Σ = covmat(gp.covfunc, x, x, hyp)

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
        dnlml[i] = sum(sum(Q.*partial_covmat(gp.covfunc, x, x, i, hyp)))/2
    end

    # return negative log marginal likelihood and its gradient
    return nlml, dnlml
end

# this new train function uses gpml matlab codes with MATALB julia module
include("gpml.jl")
function train(gp::GaussianProcess, x, y, iter=1000, verbose=true)
    hyp = minimize_gpml(gp, x, y, -iter, verbose)
    return hyp
end

function train2(gp::GaussianProcess, x, y, iter=1000, verbose=true)
    hyp, evals, iters = minimize(gethyp(gp.covfunc), lik, -iter, gp, x, y)
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

function sample(gp::GaussianProcess, x)
    n = size(x, 1)
    y = zeros(n)
    idx = [];
    for i in randperm(n)
        if isempty(idx)
            idx = [i];
            m = meanvec(gp.meanfunc, x[[i]])
            s2 = covmat(gp.covfunc, x[[i]], x[[i]])
        else
            m, s2 = predict(gp, x[idx], y[idx], x[[i]])
            idx = vcat(idx, i)
        end
        y[i] = m[1] + sqrt(s2[1])*randn()
    end
    return y
end

export GaussianProcess,
       lik, train, train!, predict, test, sample
