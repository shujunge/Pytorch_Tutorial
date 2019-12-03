#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

cv::Mat tensorToMat(const at::Tensor &one_heat_map)
{ 
  //one_heat_map pytorch tensor:(H,W)
  cv::Mat result(one_heat_map.size(0), one_heat_map.size(1), CV_8UC1);
 
  for( int i = 0; i < one_heat_map.size(0); ++i )
 {
  for(int j=0;j<one_heat_map.size(1);j++)
  {
    result.at<char>(i,j)=one_heat_map[i][j].item<int>();
  }
 }
  return result;
}

/* main */
int main(int argc, const char* argv[]) 
{
    std::cout << "###################################################################\n";
    std::cout << "Starting c++ inference.....\n";
    if (argc < 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> "
      << "<path-to-image> \n";
    return -1;
    }

    torch::jit::script::Module module;
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }
    
    std::cout << "c++ load model ok\n";

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 512, 512}));

    // evalute time
    double t = (double)cv::getTickCount();
    module.forward(inputs).toTensor();
    t = (double)cv::getTickCount() - t;
    printf("execution time = %gs\n", t / cv::getTickFrequency());
    inputs.pop_back();

    // load image with opencv
    cv::Mat image;
    image = cv::imread(argv[2]);
    cv::cvtColor(image, image, CV_BGR2RGB);

    cv::Mat img_float;
    cv::resize(image, img_float, cv::Size(512, 512));

    at::Tensor img_tensor = torch::from_blob(img_float.data, {1, img_float.rows, img_float.cols, 3}, at::kByte);
    img_tensor = img_tensor.to(at::kFloat);

    //std::cout<<img_tensor[0].slice(2,0,1).slice(/*dim=*/0, /*start=*/0, /*end=*/6).slice(/*dim=*/1, /*start=*/0, /*end=*/6)<<std::endl;


    // transform image
    img_tensor = img_tensor.permute({0,3,1,2});
    img_tensor = img_tensor/255.0;
    img_tensor[0][0] = (img_tensor[0][0] - 0.485) / 0.229;
    img_tensor[0][1] = (img_tensor[0][1] - 0.456) / 0.224;
    img_tensor[0][2] = (img_tensor[0][2] - 0.406) / 0.225;

    inputs.push_back(img_tensor);
//    std::cout<<img_tensor[0].slice(0,0,1).slice(/*dim=*/1, /*start=*/0, /*end=*/6).slice(/*dim=*/2, /*start=*/0, /*end=*/6)<<std::endl;

    
    // Execute the model and turn its output into a tensor.
    at::Tensor out_tensor = module.forward(inputs).toTensor();
    out_tensor = out_tensor.argmax(1).squeeze().detach();
    std::cout << "out_tensor results:"<<out_tensor.sizes() << '\n';

    //std::cout << "C++ inference results:"<<out_tensor<< '\n';
    std::cout << "the last results:"<<out_tensor[511][511].item<int>()<< '\n';
    cv::Mat results = tensorToMat(out_tensor);
    cv::imwrite("results/c++_infernce.png",results);
    return 0;
}

