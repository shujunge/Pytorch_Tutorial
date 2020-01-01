//
// Created by zf on 19-12-30.
//

#ifndef MY_MODEL_H
#define MY_MODEL_H


#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>



// The batch size for training.
const int64_t kTrainBatchSize = 64;
// The batch size for testing.
const int64_t kTestBatchSize = 1000;
// The number of epochs to train.
const int64_t kNumberOfEpochs = 2;
// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 2;

namespace  modelzoo
{
    struct VGGNetImpl: torch::nn::Module {
        // VGG-16 Layer
        // conv1_1 - conv1_2 - pool 1 - conv2_1 - conv2_2 - pool 2 - conv3_1 - conv3_2 - conv3_3 - pool 3 -
        // conv4_1 - conv4_2 - conv4_3 - pool 4 - conv5_1 - conv5_2 - conv5_3 - pool 5 - fc6 - fc7 - fc8
        VGGNetImpl(): conv1_1(torch::nn::Conv2dOptions(1, 10, 3).padding(1)),
                   conv1_2(torch::nn::Conv2dOptions(10, 20, 3).padding(1)),

                   conv2_1(torch::nn::Conv2dOptions(20, 30, 3).padding(1)),
                   conv2_2(torch::nn::Conv2dOptions(30, 40, 3).padding(1)),

                   conv3_1(torch::nn::Conv2dOptions(40, 50, 3).padding(1)),
                   conv3_2(torch::nn::Conv2dOptions(50, 60, 3).padding(1)),
                   conv3_3(torch::nn::Conv2dOptions(60, 64, 3).padding(1)),

                   conv4_1(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
                   conv4_2(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
                   conv4_3(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),

                   conv5_1(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
                   conv5_2(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
                   conv5_3(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
                   fc1(128*3*3, 64),
                   fc2(64, 10)
        {
            register_module("conv1_1", conv1_1);
            register_module("conv1_2", conv1_2);

            register_module("conv2_1", conv2_1);
            register_module("conv2_2", conv2_2);

            register_module("conv3_1", conv3_1);
            register_module("conv3_2", conv3_2);
            register_module("conv3_3", conv3_3);


            register_module("conv4_1", conv4_1);
            register_module("conv4_2", conv4_2);
            register_module("conv4_3", conv4_3);


            register_module("conv5_1", conv5_1);
            register_module("conv5_2", conv5_2);
            register_module("conv5_3", conv5_3);

            register_module("fc1", fc1);
            register_module("fc2", fc2);

        }

        // Implement Algorithm
        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(conv1_1->forward(x));
            x = torch::relu(conv1_2->forward(x));
//        x = torch::max_pool2d(x, 2);

            x = torch::relu(conv2_1->forward(x));
            x = torch::relu(conv2_2->forward(x));
            x = torch::max_pool2d(x, 2);

            x = torch::relu(conv3_1->forward(x));
            x = torch::relu(conv3_2->forward(x));
            x = torch::relu(conv3_3->forward(x));
            x = torch::max_pool2d(x, 2);

            x = torch::relu(conv4_1->forward(x));
            x = torch::relu(conv4_2->forward(x));
            x = torch::relu(conv4_3->forward(x));
//        x = torch::max_pool2d(x, 2);

            x = torch::relu(conv5_1->forward(x));
            x = torch::relu(conv5_2->forward(x));
            x = torch::relu(conv5_3->forward(x));
            x = torch::max_pool2d(x, 2);


            x = x.view({-1, 128*3*3});

            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);

            return torch::log_softmax(x, 1);
        }

        // Declare layers
        torch::nn::Conv2d conv1_1;
        torch::nn::Conv2d conv1_2;
        torch::nn::Conv2d conv2_1;
        torch::nn::Conv2d conv2_2;
        torch::nn::Conv2d conv3_1;
        torch::nn::Conv2d conv3_2;
        torch::nn::Conv2d conv3_3;
        torch::nn::Conv2d conv4_1;
        torch::nn::Conv2d conv4_2;
        torch::nn::Conv2d conv4_3;
        torch::nn::Conv2d conv5_1;
        torch::nn::Conv2d conv5_2;
        torch::nn::Conv2d conv5_3;

        torch::nn::Linear fc1, fc2;
    };

    TORCH_MODULE(VGGNet);//to save and load  model


    struct LNetImpl : torch::nn::Module {
        LNetImpl()
                : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
                  conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
                  fc1(320, 50),
                  fc2(50, 10) {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv2_drop", conv2_drop);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
            x = torch::relu(
                    torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
            x = x.view({-1, 320});
            x = torch::relu(fc1->forward(x));
            x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
            x = fc2->forward(x);
            return torch::log_softmax(x, /*dim=*/1);
        }

        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::FeatureDropout conv2_drop;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
    };
    TORCH_MODULE(LNet); //to save and load  model


}


template <class T, typename DataLoader>
void train(size_t epoch, T& model, torch::Device device, DataLoader& data_loader,
           torch::optim::Optimizer& optimizer, size_t dataset_size) {
    model->train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % kLogInterval == 0) {
            std::printf(
                    "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
        }
    }
}

template <class T, typename DataLoader>
void test(T& model, torch::Device device, DataLoader& data_loader, size_t dataset_size) {
    torch::NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model->forward(data);
        test_loss += torch::nll_loss(
                output,
                targets,
                /*weight=*/{},
                Reduction::Sum)
                .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / dataset_size);
}


#endif //MY_MODEL_H
