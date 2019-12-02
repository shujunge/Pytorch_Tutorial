#include <torch/torch.h>
#include <iostream>

using namespace std;
using namespace torch;
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

int main()
{
    Net net(4, 5);
    torch::Device device(torch::kCPU, 0);//torch::kCUDA  and torch::kCPU
    for (const auto& pair : net.named_parameters())
    {
        cout << pair.key() << ": " << pair.value() << endl;
    }
    net.to(device);
    at::Tensor output = net.forward(torch::ones({2, 4}).to(device));
    cout<<output<<endl;
    return 0;
}

