function shuffle!(a::Vector)::typeof(a)
    l = length(a);
    for i in 1:l
        ri = rand(1:l)
        t = a[ri];
        a[ri] = a[i]
        a[i] = t;
    end
    return a;
end
