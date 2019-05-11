function accumulative_backpropagation!(net, train_datum::TestDatum,
    dCx_dw_buffer, dCx_db_buffer, dCx_da_buffer, Bj_buffer)

    #feedforward
    feedforward!(net, train_datum);

    #BackPropagate
    update_Bj_buffer!(net, Bj_buffer, dCx_da_buffer, train_datum);
    accumulate_dCx_dw_buffer!(net, dCx_dw_buffer, Bj_buffer);
    accumulate_dCx_db_buffer!(net, dCx_db_buffer, Bj_buffer);

end

function update_Bj_buffer!(net, Bj_buffer, dCx_da_buffer, train_datum)

    #Reset Buffers
    for l in 2:net.depth
        Bj_buffer[l] .= 0.0;
        dCx_da_buffer[l] .= 0.0;
    end

    for l in net.depth:-1:2
        if l == net.depth
            for j in 1:outlength(net)
                dCx_da_buffer[net.depth][j] = single_cost_prime(output(net)[j], output(train_datum)[j]);
                Bj_buffer[net.depth][j] = act_function_prime(net.z[net.depth][j]) * dCx_da_buffer[net.depth][j];
            end
        else
            nextl = l + 1;
            for k in 1:net.lsizes[l]
                for j in 1:net.lsizes[nextl]
                    dCx_da_buffer[l][k] += net.w[nextl][j,k] * Bj_buffer[nextl][j];
                end
                Bj_buffer[l][k] = act_function_prime(net.z[l][k]) * dCx_da_buffer[l][k];
            end
        end
    end
end

function accumulate_dCx_db_buffer!(net, dCx_db_buffer, Bj_buffer)
    for l in 2:net.depth
        for j in 1:net.lsizes[l]
            dCx_db_buffer[l][j] += Bj_buffer[l][j];
        end
    end
end

function accumulate_dCx_dw_buffer!(net, dCx_dw_buffer, Bj_buffer)
    for l in 2:net.depth
        prevl = l-1;
        for j in 1:net.lsizes[l]
            for k in 1:net.lsizes[prevl]
                dCx_dw_buffer[l][j,k] += net.a[prevl][k] * Bj_buffer[l][j];
            end
        end
    end
end
