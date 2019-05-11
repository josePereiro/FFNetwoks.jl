struct Network

    lsizes::Tuple{Vararg{Int}};
    b::Tuple{Vararg{Vector{Float64}}};
    w::Tuple{Vararg{Matrix{Float64}}};
    z::Tuple{Vararg{Vector{Float64}}};
    a::Tuple{Vararg{Vector{Float64}}};
    depth::Int;
    inlength::Int;
    outlength::Int;

    function Network(lsizes::Tuple{Vararg{Int}})
        @assert length(lsizes) > 0 && all(lsizes .> 0)

        #Initializing the bias randomly
        #For the first layer the bias is not usable.
        b = Vector{Vector{Float64}}();
        push!(b,zeros(1));
        for l in 2:length(lsizes)
            push!(b,randn(lsizes[l]))
        end

        #Initializing the ws
        #For the first layer the weight is not usable.
        w = Vector{Matrix{Float64}}();
        push!(w, zeros(1,1));
        for l in 2:length(lsizes)
            push!(w,randn(lsizes[l],lsizes[l - 1]))
        end

        #The network store the values z and a of the last feedforward
        z = Vector{Vector{Float64}}();
        a = Vector{Vector{Float64}}();
        for l in 1:length(lsizes)
            push!(z, zeros(lsizes[l]));
            push!(a, zeros(lsizes[l]));
        end

        #Constructing
        new(lsizes, tuple(b...), tuple(w...), tuple(z...), tuple(a...),
            length(lsizes), first(lsizes), last(lsizes))
    end
end
Network(lsizes::Int...) = Network(lsizes);
Network(lsizes::Vector{Int}) = Network(lsizes...);
Base.size(net::Network) = net.lsizes;
Base.size(net::Network, l::Int) = net.lsizes[l];
output(net::Network) = last(net.a);
input(net::Network) = first(net.a);
outlength(net::Network) = length(output(net));
inlength(net::Network) = length(input(net));

function feedforward!(net::Network, input::Vector{Float64})::Vector{Float64}
    @assert net.inlength == length(input)

    #first layer
    net.z[1] .= input;
    net.a[1] .= input;

    #Other layers
    for l in 2:net.depth
        net.z[l] .= net.w[l]*net.a[l-1] + net.b[l];
        net.a[l] .= act_function.(net.z[l]);
    end

    #The output
    return output(net);
end

feedforward!(net::Network, testdata) = feedforward!(net, input(testdata));
