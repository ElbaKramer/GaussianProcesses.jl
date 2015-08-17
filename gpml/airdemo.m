air = csvread('../data/air.csv', 1, 0);
x = air(:,1);
y = air(:,2);
inference = @infDelta;
meanfunc = @meanZero;
covfunc = {@covSum, {@covSEiso, @covNoise}};
likfunc = @likDelta;
hyp.mean = [];
hyp.cov = zeros(eval(feval(covfunc{:})), 1);
hyp.lik = [];
[nlml, dnlml] = gp(hyp, inference, meanfunc, covfunc, likfunc, x, y);
opt = minimize(hyp, @gp, -500, inference, meanfunc, covfunc, likfunc, x, y);
pred.mean = [];
pred.cov = opt.cov(1:2);
pred.lik = opt.cov(3);
[m, s2] = gp(pred, @infExact, meanfunc, @covSEiso, @likGauss, x, y, x);
plot(x,y)
hold on
plot(x,m)