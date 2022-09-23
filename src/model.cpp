#include "model.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "kl_error.hpp"
#include "math.h"

float softmax_genuine(float* out_arr){
    float log_sum = 0.f;
    for(int i =0 ; i < 3; i++){
        out_arr[i] = std::exp(out_arr[i]);
        log_sum += out_arr[i];
    }
    return out_arr[1] / log_sum;
}

KLError Model::init(const char *model_path)
{
    try
    {
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
    }
    catch (...)
    {
        return KLError::MODEL_LOAD_ERROR;
    }
    return KLError::NONE;
}

float Model::inference(const char *img_path)
{
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
    float *model_input = interpreter_->typed_input_tensor<float>(0);
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // float *new_img_addr;
    // this->convert_image(image, new_img_addr);
    // fprintf(stdout, "w = %d, h = %d, c = %d \n", image.rows, image.cols, image.channels());
    // fprintf(stdout,"new_img_addr[last]= %f\n",*(new_img_addr+image.rows * image.cols * image.channels() -1));
    int dims[] = {3, image.rows, image.cols};
    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);
    cv::Mat chw(3, dims, CV_32FC1);
    std::vector<cv::Mat> planes = {
        cv::Mat(image.rows, image.cols, CV_32F, chw.ptr<float>(0)), // swap 0 and 2 and you can avoid the bgr->rgb conversion !
        cv::Mat(image.rows, image.cols, CV_32F, chw.ptr<float>(1)),
        cv::Mat(image.rows, image.cols, CV_32F, chw.ptr<float>(2))};
    cv::split(image_f, planes);

    memcpy(model_input, chw.ptr<float>(0), chw.total() * sizeof(float));
    interpreter_->Invoke();
    float *output = interpreter_->typed_output_tensor<float>(0);
    return softmax_genuine(output);
    // return 0.0;
}

void Model::convert_image(const cv::Mat &src, float *dest)
{
    if(src.empty())
    {
        fprintf(stderr, "Failed to load image\n");
        exit(-1);
    }
    fprintf(stdout,"%p", (void*)dest);
}
