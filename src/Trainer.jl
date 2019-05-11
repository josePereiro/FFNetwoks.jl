function train!(net::Network, train_data::Vector{TestDatum},
        test_data::Vector{TestDatum}, epoch::Int,
        mini_batch_size::Int, eta::Float64; verbose = false)
    @assert all((length(train_data), length(test_data)) .> 1);
    @assert all(inlength(net) .== inlength.(train_data))
    @assert all(outlength(net) .== outlength.(train_data))
    @assert all(inlength(net) .== inlength.(test_data))
    @assert all(outlength(net) .== outlength.(test_data))

    #Gradient Buffers
    #For the first layer the gradients are not usable.
    dCx_dw_buffer::NTuple{net.depth, Matrix{Float64}} =
        tuple(zeros(Float64,1,1),[Matrix{Float64}(undef, net.lsizes[l], net.lsizes[l-1]) for l in 2:net.depth]...);
    dCx_db_buffer::NTuple{net.depth, Vector{Float64}} =
        tuple(zeros(Float64,1),[Vector{Float64}(undef,n) for n in net.lsizes[2:end]]...);

    #Extra Buffers
    #Store the gradients of dC/da for all layers...
    #For the first layer the value are not usable.
    dCx_da_buffer::NTuple{net.depth, Vector{Float64}} = tuple(zeros(Float64,1),[Vector{Float64}(undef,n) for n in net.lsizes[2:end]]...)
    #Will Store a partial result that is redundant.
    #For the first layer the value are not usable.
    Bj_buffer::NTuple{net.depth, Vector{Float64}} = tuple(zeros(Float64,1),[Vector{Float64}(undef,n) for n in net.lsizes[2:end]]...);

    #Epoch
    if verbose
        acost::Float64 = 0.0;
        bcost::Float64 = 0.0;
        startime::Float64 = 0.0;

    end
    for e in 1:epoch

        #Info
        if verbose
            printstyled("Epoch $e"; color = :greev);
            println();
            startime = time();
            if e == 1
                bcost = cost!(net, test_data);
            else
                bcost = acost;
            end
            println("Cost Before $bcost");
        end

        #Unsorting data
        shuffle!(train_data);

        #Mini batching
        train_subset_length = min(mini_batch_size, length(train_data));
        for tdi in train_subset_length:train_subset_length:length(train_data)

            #Mini_batch
            train_subset = train_data[tdi - train_subset_length + 1:tdi];

            #Updating gradients for this mini batch
            update_gradients!(net, train_subset, dCx_dw_buffer, dCx_db_buffer, dCx_da_buffer, Bj_buffer);

            #appliying gradient desendence
            update_ws!(net, dCx_dw_buffer, eta);
            update_bs!(net, dCx_db_buffer, eta);

        end # mini batch train

        #Evaluate Info
        if verbose
            acost = cost!(net, test_data);
            print("Cost After  $acost");
            if acost - bcost < 0
                printstyled(" ($(acost - bcost))\n"; color = :blue, bold = true);
            else
                printstyled(" ($(acost - bcost))\n"; color = :red, bold = true);
            end
            println("Accuracy $(evaluate!(net,test_data))");
            println("Time $(round(Int,time() - startime)) seconds")
            println();
        end

    end #Epoch

end

function update_gradients!(net, train_data, dCx_dw_buffer, dCx_db_buffer, dCx_da_buffer, Bj_buffer)

    #Reset gradients to 0.0
    for l in 1:net.depth
        dCx_dw_buffer[l] .= 0.0;
        dCx_da_buffer[l] .= 0.0;
    end

    #Accumulate gradients for each testdata
    for train_datum in train_data
        accumulative_backpropagation!(net, train_datum, dCx_dw_buffer, dCx_db_buffer, dCx_da_buffer, Bj_buffer);
    end

    #Averaging
    average_grads!(net, dCx_dw_buffer,dCx_db_buffer, length(train_data));

end

function average_grads!(net, dCx_dw_buffer, dCx_db_buffer, train_set_length)
    for l in 2:net.depth
        prevl = l - 1;
        for j in 1:net.lsizes[l]
            for k in 1:net.lsizes[prevl]
                dCx_dw_buffer[l][j,k] = dCx_dw_buffer[l][j,k] / train_set_length;
            end
            dCx_db_buffer[l][j] = dCx_db_buffer[l][j] / train_set_length;
        end
    end
end

function update_ws!(net, dCx_dw_buffer, eta)
    for l in 2:net.depth
        prevl = l -1;
        for j in 1:net.lsizes[l]
            for k in 1:net.lsizes[prevl]
                net.w[l][j,k] -= eta * dCx_dw_buffer[l][j,k];
            end
        end
    end
end

function update_bs!(net, dCx_db_buffer, eta)
    for l in 2:net.depth
        for j in 1:net.lsizes[l]
            net.b[l][j] -= eta * dCx_db_buffer[l][j];
        end
    end
end

function evaluate!(net, test_data)
    good_answers = 0.0;
    for test_datum in test_data
        expout = output(test_datum);
        netout = feedforward!(net, input(test_datum));

        if findmax(expout)[2] == findmax(netout)[2]
           good_answers = good_answers + 1;
        end
    end

    return good_answers/length(test_data);
end
