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
