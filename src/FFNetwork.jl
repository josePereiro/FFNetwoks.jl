module FFNetwork

    include("ActFunction.jl")
    include("TestDatum.jl")
    include("Network.jl")
    include("Cost.jl")
    include("Tools.jl")
    include("BackPropagation.jl")
    include("Trainer.jl")
    include("MNIST.jl")


    export Network;
    export TestDatum;
    export input;
    export output;
    export inlength;
    export outlength;
    export feedforward!;
    export train!;

end # module
