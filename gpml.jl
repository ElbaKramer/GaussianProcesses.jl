using MATLAB

function gpml(f::CovarianceFunction)
    if typeof(f) <: SimpleCovarianceFunction
        covfunc = string("{@", f.fname, "}")
    elseif typeof(f) <: CompositeCovarianceFunction
        operands = join([gpml(ff) for ff in f.fvec], ",")
        covfunc = string("{@", f.fname, ",{", operands, "}}")
    end
    return covfunc
end

function minimize(gp::GaussianProcess, x, y, iter, verbose)
    if verbose
        s = MSession()
    else
        s = MSession(0)
    end
    eval_string(s, "
    run('gpml/startup');
    inference = @infDelta;
    meanfunc = @meanZero; hyp.mean = [];
    likfunc = @likDelta; hyp.lik = [];")
    covfunc = gpml(gp.covfunc)
    eval_string(s, "covfunc = $covfunc;")
    hyp = mxarray(gethyp(gp.covfunc))
    put_variable(s, :hypcov, mxarray(hyp))
    eval_string(s, "hyp.cov = hypcov;")
    put_variable(s, :x, mxarray(x))
    put_variable(s, :y, mxarray(y))
    put_variable(s, :iter, iter)
    eval_string(s, "hyp = minimize(hyp, @gp, -iter, inference, meanfunc, covfunc, likfunc, x, y);")
    eval_string(s, "hypcov = hyp.cov;")
    hyp = jarray(get_mvariable(s, :hypcov))
    close(s)
    return vec(hyp)
end
