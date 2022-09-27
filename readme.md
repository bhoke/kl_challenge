## Keyless Challenge

Your task is to convert a PyTorch model to TFLite and reproduce the same inference scores obtained in python in this C++ project. For this challenge, we will use a computer vision deep learning model for anti-spoofing from the [Silent-Face-Anti-Spoofing github repo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

The above repo contains two deep learning models inside the [resources/anti_spoof_models](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models) folder, but we are only interested in the `2.7_80x80_MiniFASNetV2.pth`. The other should be ignored.

This C++ project is organized as follows:

- `assets/models` is the folder where you should place your converted TFLite model, namely `2.7_80x80_MiniFASNetV2.tflite`.
- `bin` contains some shell scripts to help you build and test the project. Both macOS and linux are supported (Big Sur 11.2.3 and Ubuntu 20.04 have been tested).
- `build/{OS_NAME}` is the folder in which the project will be built.
- `cmake` contains a CMake helper file required when building for linux.
- `src` contains the source code you will work on.
- `test` contains test code and a fixtures folder where you should place your test images.
- `vendors` contains external dependencies (opencv2 and tflite) required by the project.

To better organize the challenge and provide you with some guidance, we will break the task into smaller ones. 

Whenever you find a question, it should be answered in place. All questions are there for a purpose.

### 1. Building the project

- Prepare your environment with CMake, a C++ 17 compiler and pthread. As far as I can remember, these are the only requirements.
- You can build the project by running `./bin/build_macos.sh` or `./bin/build_linux.sh`, but it will probably fail. What is the cause?
  - I have worked in Linux environment. The cause is that OpenCV libraries (including dependencies) are not linked to our program. I added them to `CMakeLists` and `generate_ar_input_file.cmake`
- Edit the `src/CMakeLists.txt` - and the `cmake/generate_ar_input_file.cmake` if you are on linux - to fix it.

### 2. Understanding Silent-Face-Anti-Spoofing

- You should start from the `test.py` file.
- Take a look at the networks' architecture. What is the last layer?
  - Last layer is linear (fully connected) layer which has shape `(128,3)` while building the model. However, softmax function is also called while preciting an image.  
- What preprocessing operations an image undergoes before being inputted to the network?
  - Image is cropped to size `(80, 80, 3)` with the help of face detector model.
  - NumPy array is converted to the `torch.Tensor`
  - `(H x W x C)` format is converted to the `(C x H x W)` format. 
- Does the input image have channel-first or channel-last format?
  - All the input imagess are channel-last format, but model has channel-first format
- What is the input image colorspace?
  - Input is read BGR colorspace and kept as is.
- How many classes image can be classified into?
  - There are 3 classes as the model output
- What is the index for the genuine (real face) classification score?
  - Genuine index is 1.
- Apart from the anti-spoofing models, does the code use any other ML model?
  - Code uses Face detection model to crop the face area from the images.

### 3. Testing Silent-Face-Anti-Spoofing

- [x] Modify the `test.py` script to only use the `2.7_80x80_MiniFASNetV2.pth` model.
- [x] Modify the `test.py` script to output only the genuine score.
- [x] Run the `test.py` script for `image_F1.jpg`, `image_F2.jpg` and `image_T1.jpg` images.
- [x] What are the genuine scores for each one of them?
  - [x] Genuine scores for `image_F1.jpg`, `image_F2.jpg` and `image_T1.jpg` are `0.09519024`, `0.00132873` and `0.903481` respectively.
- [x] You will have to reproduce the scores from the previous step later when using TFLite.

### 4. Converting the model to TFLite

- Is it possible to convert the model directly from PyTorch to TFLite?
  - Technically, it is possible to transfer all the PyTorch weights to a Keras/TF model and converting created model into TFLite is possible (I have written similar library for CaffeTF). However, it is really tedious job for most cases.
- If not, which are the intermediates required for this conversion?
  - PyTorch has native support to ONNX and ONNX provides many deployment options including TensorFlow. I have used ONNX for this conversion.
- [x] Convert the `2.7_80x80_MiniFASNetV2.pth` model to TFLite and place it inside `assets/models`.

### 5. Generating test images
 
- [x] You should generate the test images for your C++ code and place them inside `test/fixtures`.
- Note from `test/test_main.cpp` that they must be in `bmp` format. Any hunch why we use `bmp` instead of `jpg` here?
  - `bmp` is an uncompressed format. I guess, when we save it as `jpeg` the pixel values may change. As a result, we could not observe exact same result. 
### 6. Implementing the C++ code

- Your task here is to complete the `model.cpp` file. There are 3 methods to be implemented:
  - `init`, which should load the TFLite model from disk and make it ready to be used.
  - `inference`, which should load an image from disk and pass it through the network.
  - `convert_image`, which should convert the image to the correct format before sending it to the network.
- The opencv2 lib is available for you to read image files from disk.
  - Does the input image have channel-first or channel-last format?
    - Input image has channel-last format, while model has channel-first format.
  - What is the input image colorspace?
    - It has BGR format.

### 7. Testing your solution

- [x] Build your solution
- [x] Test your solution by running `bin/test_macos.sh` or `bin/test_linux.sh`
- What are the genuine scores for each test image?
  - Genuine scores for `image_F1.bmp`, `image_F2.bmp` and `image_T1.bmp` are `0.09519024`, `0.00132873` and `0.903481` respectively, which are same with original model.
