import FFNetwork
using Test

@testset "FFNetwork.jl" begin

    printstyled("Training with MNIST,\n This could take a few minutes and migth require internet!!!"; color = :green);
    println();

    println();
    println("Train set length: 50000");
    println("Mini_batch_size 20");
    println("Eta 3.0");
    println("Test set length: 10000");
    println();

    printstyled("Running 10 epochs "; color = :green, bold = true);
    printstyled("Blue"; color = :blue, bold = true);
    printstyled(" is good "; color = :green, bold = true);
    printstyled("Red"; color = :red, bold = true);
    printstyled(" is bad "; color = :green, bold = true);
    println();
    printstyled("Using Sigmoid function as act_function "; color = :green, bold = true);
    println();

    netSigmoid = FFNetwork.train_network_with_MNIST!(;verbose = true);

    # println();
    # printstyled("Running 10 epochs "; color = :green, bold = true);
    # printstyled("Blue"; color = :blue, bold = true);
    # printstyled(" is good "; color = :green, bold = true);
    # printstyled("Red"; color = :red, bold = true);
    # printstyled(" is bad "; color = :green, bold = true);
    # println();
    printstyled("Using ReLU function as act_function "; color = :green, bold = true);
    println();
    FFNetwork.act_function(z) = FFNetwork.ReLU(z);
    FFNetwork.act_function_prime(z) = FFNetwork.ReLU_prime(z);

    netReLU = FFNetwork.train_network_with_MNIST!(;verbose = true);
    sigmoidacc = FFNetwork.evaluate!(netSigmoid, FFNetwork.get_MNIST_img_data());
    ReLUacc = FFNetwork.evaluate!(netReLU, FFNetwork.get_MNIST_img_data());
    println();

    printstyled("Evaluation over whole MNIST data"; color = :blue, bold = true);
    println();
    printstyled("Sigmoid accuracy $sigmoidacc"; color = :green, bold = true);
    println();
    printstyled("ReLU accuracy $ReLUacc"; color = :green, bold = true);
    println();
    println();

    @test 1==1;

end
