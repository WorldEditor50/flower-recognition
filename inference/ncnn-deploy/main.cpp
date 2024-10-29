#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "resnet18.h"

int main()
{
    Resnet18 model;
    /* load model */
    if (!model.load("./model/flower_resnet18.ncnn")) {
        std::cout<<"failed to load model."<<std::endl;
        return -1;
    }
    /* load image */
    cv::Mat img = cv::imread("./sunflower.jpeg");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return -1;
    }
    /* classify */
    int i = model(img);
    if (i == 0) {
        std::cout<<"daisy"<<std::endl;
    } else if (i == 1) {
        std::cout<<"dandelion"<<std::endl;
    } else if ( i == 2) {
        std::cout<<"rose"<<std::endl;
    } else if ( i == 3) {
        std::cout<<"sunflower"<<std::endl;
    } else if ( i == 4) {
        std::cout<<"tulip"<<std::endl;
    } else {
        std::cout<<"unknow"<<std::endl;
    }
    return 0;
}
