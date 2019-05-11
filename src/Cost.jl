
"""
    The function that compute the cost of a single output value.
"""
single_cost(netout, expout) = ((netout - expout)^2)/2;
"""
    The prime derivative of the single_cost function.
"""
single_cost_prime(netout, expout) = (netout - expout);

function cost(net::Network, testdatum::TestDatum)
    @assert outlength(net) == outlength(testdatum);
    @assert inlength(net) == inlength(testdatum);
    return sum(single_cost.(output(net),output(testdatum)));
end

function cost!(net::Network, testdatum::TestDatum)
    feedforward!(net,input(testdatum));
    return cost(net,testdatum);
end

function cost!(net::Network, testdata::Vector{TestDatum})
    costsum = 0.0;
    for test in testdata
        costsum += cost!(net,test);
    end
    return costsum/length(testdata);
end
