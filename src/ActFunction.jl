"""
    The function that will modulate tha activation value of
    each non-input neuron in the network. It maps the z vale (∑wx + b)
    of the neurons to ℝ. By default its taken the sigmoid function
    one(z)/(one(z) + exp(-z)). To change it, just redefine it!!!
    But remember change the act_function_prime function too!!!!!!
"""
function act_function(z::T)::T where T<:Real
    return sigmoid(z);
end

"""
    The first derivative of the act_function.
"""
function act_function_prime(z::T) where T<:Real
    return sigmoid_prime(z);
end

function sigmoid(z::T)::T where T<:Real
    return one(z)/(one(z) + exp(-z));
end
function sigmoid_prime(z::T) where T<:Real
    return exp(-z)/((one(z) + exp(-z))^2);
end

ReLU_ng = 0.1;

function ReLU(z::T)::T where T<:Real
    return max(ReLU_ng * z, z);
end

function ReLU_prime(z::T)::T where T<:Real
    if z < 0
        return ReLU_ng;
    else
        return one(z);
    end
end
