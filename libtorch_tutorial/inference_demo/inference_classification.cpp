#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* main */
int main(int argc, const char* argv[]) 
{
    std::cout << "###################################################################\n";
    std::cout << "Starting c++ inference.....\n";
    if (argc < 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module> "
      << "<path-to-image>  <path-to-category-text>\n";
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
    inputs.push_back(torch::rand({64, 3, 224, 224}));

    // evalute time
    double t = (double)cv::getTickCount();
    module.forward(inputs).toTensor();
    t = (double)cv::getTickCount() - t;
    printf("execution time = %gs\n", t / cv::getTickFrequency());
    inputs.pop_back();

    // load image with opencv
    cv::Mat image;
    image = cv::imread(argv[2], 1);
    cv::cvtColor(image, image, CV_BGR2RGB);

    cv::Mat img_float;
    cv::resize(image, img_float, cv::Size(224, 224));

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
    std::cout << "C++ inference results:"<<out_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

    // Load labels
    std::string label_file = argv[3];
    std::ifstream rf(label_file.c_str());
    CHECK(rf) << "Unable to open labels file " << label_file;
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(rf, line))
        labels.push_back(line);

    // print predicted top-5 labels
    std::tuple<torch::Tensor,torch::Tensor> result = out_tensor.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

    auto top_scores_a = top_scores.accessor<float,1>();
    auto top_idxs_a = top_idxs.accessor<int,1>();

    for (int i = 0; i < 5; ++i)
    {
        int idx = top_idxs_a[i];
        std::cout << "top-" << i+1 << " label: ";
        std::cout << labels[idx] << ", score: " << top_scores_a[i] << std::endl;
    }

    return 0;
}

