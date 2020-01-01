//
// Created by zf on 19-12-1.
//
#include "my_model.h"


using namespace modelzoo;

// Where to find the MNIST dataset.
const char* DataRoot = "./data";


auto main() -> int
{
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    LNet model;

    model->to(device);


    /************data loader************************/

    auto train_dataset = torch::data::datasets::MNIST(DataRoot)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(
            DataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();


    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(train_dataset), kTrainBatchSize);
    auto  test_loader =
            torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);


    for (auto& batch : *train_loader)
    {
        at::Tensor data = batch.data.to(device);
        at::Tensor targets = batch.target.to(device);

        std::cout<< "images:"<< data.sizes()<<" ";
        auto imag= data.slice(0,0,1).slice(1, 0, 1).squeeze().detach(); //[28,28]
        std::cout<< "results:"<< imag.sizes()<<" ";
        std::cout<< "targets:"<< targets.sizes()<<std::endl;
        return 0;
    }
    /************************************/


    torch::optim::SGD optimizer(
            model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *test_loader, test_dataset_size);
    }

    torch::save(model, "model.pt");
    torch::load(model, "model.pt");

    //print model parameters
    for (const auto& pair : model->named_parameters()) {
        std::cout << pair.key() << ": " << pair.value().sizes() << std::endl;
    }
    test(model, device, *test_loader, test_dataset_size);

}
