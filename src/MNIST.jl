using Flux.Data.MNIST;

function loadMNIST()
    global MNISTimgs = MNIST.images();
    global MNISTlabels = MNIST.labels();
end

struct MNISTImg
    dbindex::Int;#The index of this image in the MNIST data base
    label::Int;#The label of the image
    data::FFNetwork.TestDatum;#the input and output of the image.
end

output(img::MNISTImg) = output(img.data);
input(img::MNISTImg) = input(img.data);
outlength(img::MNISTImg) = outlength(img.data);
inlength(img::MNISTImg) = inlength(img.data);
data(img::MNISTImg) = img.data;

MNIST_img_length = 28*28;
Labels_length = 10;

labels_tuple = NTuple{10,Vector{Float64}}(
    ([1.,0.,0.,0.,0., 0.,0.,0.,0.,0.],
    [0.,1.,0.,0.,0., 0.,0.,0.,0.,0.],
    [0.,0.,1.,0.,0., 0.,0.,0.,0.,0.],
    [0.,0.,0.,1.,0., 0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,1., 0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0., 1.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0., 0.,1.,0.,0.,0.],
    [0.,0.,0.,0.,0., 0.,0.,1.,0.,0.],
    [0.,0.,0.,0.,0., 0.,0.,0.,1.,0.],
    [0.,0.,0.,0.,0., 0.,0.,0.,0.,1.]));

vectorize_label(l::Int)::Vector{Float64} = labels_tuple[l+1];

function get_MNIST_img_data(imgindex::Int)::MNISTImg
    imglabel = MNISTlabels[imgindex];
    imgvalues = reshape(Float64.(MNISTimgs[imgindex]),MNIST_img_length)
    MNISTImg(imgindex ,imglabel, TestDatum(imgvalues,vectorize_label(imglabel)));
end
function get_MNIST_img_data()::Vector{MNISTImg}
    return [MNISTImg(imgindex ,MNISTlabels[imgindex], TestDatum(reshape(Float64.(MNISTimgs[imgindex]),MNIST_img_length),vectorize_label(MNISTlabels[imgindex]))) for imgindex in 1:length(MNISTimgs)];
end

function get_MNIST_train_and_test_data()::NTuple{2,Vector{MNISTImg}}
    MNISTLength = length(MNISTlabels);
    total = shuffle!([get_MNIST_img_data(i) for i in 1:MNISTLength]);
    return (total[1:MNISTLength - 10000], total[MNISTLength - 9999:MNISTLength])
end

function train_network_with_MNIST!(net::Network;
    epoch = 10, eta = 3.0, mini_batch_size = 20, verbose = false)
    @assert outlength(net) == Labels_length;
    @assert inlength(net) == MNIST_img_length;
    loadMNIST();
    traindata, testdata = get_MNIST_train_and_test_data();
    train!(net, data.(traindata) , data.(testdata) , epoch, mini_batch_size, eta; verbose = verbose);
end

# (net::Network, train_data::Vector{TestDatum},
#         test_data::Vector{TestDatum}, epoch::Int,
#         mini_batch_size::Int, eta::Float64; verbose = false)
