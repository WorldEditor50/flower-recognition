#ifndef RESNET18_H
#define RESNET18_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <ncnn/net.h>
#include <ncnn/layer.h>
#include <string>

class Resnet18
{
private:
    bool hasLoadedModel;
    ncnn::Net net;
public:
    Resnet18(){}
    bool load(const std::string &model)
    {
        std::string paramFile = model + ".param";
        std::string modelFile = model + ".bin";
        int ret = net.load_param(paramFile.c_str());
        if (ret != 0) {
            return false;
        }
        ret = net.load_model(modelFile.c_str());
        if (ret != 0) {
            return false;
        }
        hasLoadedModel = true;
        return true;
    }

    int operator()(const cv::Mat &img)
    {
        if (!hasLoadedModel) {
            return -1;
        }
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data,
                                                      ncnn::Mat::PIXEL_BGR2RGB,
                                                      img.cols, img.rows,
                                                      224, 224);
        const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };
        const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat out;
        ex.input("in0", in);
        ex.extract("out0", out);
        ncnn::Layer *softmax = ncnn::create_layer("Softmax");
        ncnn::ParamDict pd;
        softmax->load_param(pd);
        softmax->forward_inplace(out, net.opt);
        float *outPtr = out.channel(0);
        int index = 0;
        float maxVal = outPtr[0];
        for (int i = 1; i < 5; i++) {
            if (maxVal < outPtr[i]) {
                index = i;
                maxVal = outPtr[i];
            }
        }
        return index;
    }
};

#endif // RESNET18_H
