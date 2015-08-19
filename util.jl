function sqdist(a, b)
    D, n = size(a)
    d, m = size(b)
    if d!=D
        error("Column lengths must agree")
    end
    mu = (m/(n+m))*mean(b,2) + (n/(n+m))*mean(a,2)
    a = broadcast(-, a, mu)
    b = broadcast(-, b, mu)
    C = broadcast(+, sum(a.*a,1)', broadcast(-, sum(b.*b,1), 2*a'*b))
    return C
end

function solvechol(L, B)
    return L\(L'\B)
end

function minimize(X, f, iter, varargin...)
    verbose = true

    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG/2

    if length(iter) == 2
        red = iter[2]
        len = iter[1]
    else
        len = iter
        red = 1
    end
    if len > 0
        S = "Linesearch"
    else
        S = "Function evaluation"
    end

    i = 0
    ls_failed = false
    f0, df0 = f(X, varargin...)

    if verbose
        println(S, " ", i, ";  Value ", f0)
    end

    fX = f0
    i = i + (len < 0)
    s = -df0
    d0 = -dot(s, s)
    x3 = red/(1 - d0)

    f3 = 0
    df3 = 0
    x2 = 0
    d2 = 0
    f2 = 0

    while i < abs(len)
        i = i + (len > 0)

        X0 = X
        F0 = f0
        dF0 = df0
        if len < 0
            M = MAX
        else
            M = min(MAX, -len-i)
        end

        # extrapolation
        while true
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = false
            while !success && M > 0
                try
                    M = M - 1
                    i = i + (len < 0)
                    f3, df3 = f(X+x3*s, varargin...)
                    if isnan(f3) || isinf(f3) || any(isnan(df3) | isinf(df3))
                        error(" ")
                    end
                    success = true
                catch
                    x3 = (x2+x3)/2
                end
            end
            if f3 < F0
                X0 = X+x3*s
                F0 = f3
                dF0 = df3
            end
            d3 = dot(df3, s)
            if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0
                break
            end
            x1 = x2
            f1 = f2
            d1 = d2
            x2 = x3
            f2 = f3
            d2 = d3
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            x3 = x1-d1*(x2-x1)^2/(B+sqrt(complex(B*B-A*d1*(x2-x1))))
            if isreal(x3)
                x3 = real(x3)
            end
            if !isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0
                x3 = x2*EXT
            elseif x3 > x2*EXT
                x3 = x2*EXT
            elseif x3 < x2+INT*(x2-x1)
                x3 = x2+INT*(x2-x1)
            end
        end

        # interpolation
        while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0
            if d3 > 0 || f3 > f0+x3*RHO*d0
                x4 = x3
                f4 = f3
                d4 = d3
            else
                x2 = x3
                f2 = f3
                d2 = d3
            end
            if f4 > f0
                x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2))
            else
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                x3 = x2+(sqrt(complex(B*B-A*d2*(x4-x2)^2))-B)/A
                if isreal(x3)
                    x3 = real(x3)
                end
            end
            if isnan(x3) || isinf(x3)
                x3 = (x2+x4)/2
            end
            x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2))
            f3, df3 = f(X+x3*s, varargin...)
            if f3 < F0
                X0 = X+x3*s
                F0 = f3
                dF0 = df3
            end
            M = M - 1
            i = i + (len < 0)
            d3 = dot(df3, s)
        end

        # check whether line search succeeded
        if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0
            X = X+x3*s
            f0 = f3
            fX = [fX, f0]
            if verbose
                println(S, " ", i, ";  Value ", f0)
            end
            s = (dot(df3, df3)-dot(df0, df3))/dot(df0, df0)*s - df3
            df0 = df3
            d3 = d0
            d0 = dot(df0, s)
            if d0 > 0
                s = -df0
                d0 = -dot(s, s)
            end
            x3 = x3*min(RATIO, d3/(d0-realmin()))
            ls_failed = false
        else
            X = X0
            f0 = F0
            df0 = dF0
            if ls_failed || i > abs(len)
                break
            end
            s = -df0
            d0 = -dot(s, s)
            x3 = 1/(1-d0)
            ls_failed = true
        end
    end

    # return value, evals, iterations
    return X, fX, i
end

export minimize
