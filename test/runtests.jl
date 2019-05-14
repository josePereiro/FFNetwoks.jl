import FFNetwork
using Test

@testset "FFNetwork.jl" begin

    println();
    printstyled("Training with MNIST,\n This could take a few minutes and migth require internet!!!"; color = :green);
    println();

    FFNetwork.loadMNIST();
    traindata, testdata = FFNetwork.get_MNIST_train_and_test_data();

    println();
    println("Train set length: 50000");
    println("Mini_batch_size 20");
    println("Eta 3.0");
    println("Test set length: 10000");
    println();

    println("Running 10 epochs ");
    println("Negative is Good, Positive is Bad");
    println();

    println("Using ReLU function as act_function ");
    println();
    FFNetwork.act_function(z) = FFNetwork.ReLU(z);
    FFNetwork.act_function_prime(z) = FFNetwork.ReLU_prime(z);
    netReLU = FFNetwork.Network(28*28,16,10);
    FFNetwork.train!(netReLU, FFNetwork.data.(traindata) , FFNetwork.data.(testdata) , 10 , 20, 3.0; verbose = true);

    println("Using Sigmoid function as act_function ");
    println();
    FFNetwork.act_function(z) = FFNetwork.sigmoid(z);
    FFNetwork.act_function_prime(z) = FFNetwork.sigmoid_prime(z);
    netSigmoid = FFNetwork.Network(28*28,16,10);
    FFNetwork.train!(netSigmoid, FFNetwork.data.(traindata) , FFNetwork.data.(testdata) , 10, 20, 3.0; verbose = true);

    sigmoidacc = FFNetwork.evaluate!(netSigmoid, FFNetwork.get_MNIST_img_data());
    ReLUacc = FFNetwork.evaluate!(netReLU, FFNetwork.get_MNIST_img_data());
    println();

    println("Evaluation over whole MNIST data");
    println("Sigmoid accuracy $sigmoidacc");
    println("ReLU accuracy $ReLUacc");
    println();

    @test 1==1;

end
