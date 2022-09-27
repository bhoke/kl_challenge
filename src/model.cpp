#include "model.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "kl_error.hpp"
#include "math.h"

KLError Model::init(const char *model_path)
{
    try
    {
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);

        if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk)
        {
            fprintf(stderr, "Failed to build interpreter.");
            exit(-1);
        }

        if (interpreter_->AllocateTensors() != kTfLiteOk)
        {
            fprintf(stderr, "Failed to allocate tensor\n");
            exit(-1);
        }
    }
    catch (...)
    {
        return KLError::MODEL_LOAD_ERROR;
    }
    return KLError::NONE;
}

float Model::inference(const char *img_path)
{
    float *model_input = interpreter_->typed_input_tensor<float>(0);
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    int img_size = image.rows * image.cols * image.channels();
    float *new_img_flat = new float[img_size];
    convert_image(image, new_img_flat);
    memcpy(model_input, new_img_flat, img_size * sizeof(float));
    delete[] new_img_flat;
    interpreter_->Invoke();
    float *output = interpreter_->typed_output_tensor<float>(0);
    return output[1];
}

void Model::convert_image(const cv::Mat &src, float *dest)
{
    if (src.empty())
    {
        fprintf(stderr, "Failed to load image\n");
        exit(-1);
    }
    int dims[] = {3, src.rows, src.cols};
    cv::Mat image_f;
    src.convertTo(image_f, CV_32F);
    cv::Mat chw(3, dims, CV_32FC1);
    std::vector<cv::Mat> planes = {
        cv::Mat(src.rows, src.cols, CV_32F, chw.ptr<float>(0)), // 80 x 80 x 4 bytes
        cv::Mat(src.rows, src.cols, CV_32F, chw.ptr<float>(1)),
        cv::Mat(src.rows, src.cols, CV_32F, chw.ptr<float>(2))};
    cv::split(image_f, planes);
    for(size_t i = 0; i < chw.total(); i++)
        dest[i] = chw.at<float>(i);
}
