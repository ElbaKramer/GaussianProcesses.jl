function meanzero(x, hyp)
    return zeros(size(x))
end

function partial_meanzero(x, hyp, i)
    return 0.0
end

function meanZero()
    return MeanFunction(:meanZero,
                        meanzero,
                        partial_meanzero,
                        [])
end

export meanZero
