using LinearAlgebra

function chol(A)
    return cholesky(Hermitian(A)).U
end

function eye(n, m)
    return I
end

function cell(n)
    return Array{Any,1}(undef, n)
end

function repmat(A, n, m)
    return repeat(A, outer=(n, m))
end