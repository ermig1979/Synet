Introduction
============

Synet is a small framework to inference neural network on CPU. Synet uses models previously trained by other deep neural network frameworks.

The main advantages  of Synet are:

* Synet is faster then most other DNN original frameworks (has great single thread CPU performance).
* Synet is header only, small C++ library.
* Synet has only one external dependence - [Simd Library](https://github.com/ermig1979/Simd).

Building of test applications for Linux
==============================
To build test applications you can run following bash script:

    git clone -b master --recurse-submodules -v https://github.com/ermig1979/Synet.git clone
    cd clone
    ./build.sh

And applications `test_darknet`, `test_inference_engine`, `test_onnx`, `test_precision`, `test_quantization`, `use_face_detection` will be created in directory `build`.
There is a detail description of these test applications below.

Darknet test application
========================
The test application `test_darknet` is used for [Darknet](https://github.com/pjreddie/darknet) to Synet model conversion:

    ./build/test_darknet -m convert -fm darknet_model.cfg -fw darknet_weigths.dat -sm synet_model.xml -sw synet_weigths.bin

Also it is used in order to compare performance and accuracy of Darknet and Synet frameworks.
There are several test scripts:

* For manual testing you can use `./test.sh` (in the file you have to manually uncomment unit test that you need).
* Script `./check.sh` checks correctness of all tests.
* Script `./perf.sh` measures performance of Synet compare to Darknet.


OpenVINO test application
========================
The test application `test_inference_engine` is used for [OpenVINO](https://github.com/openvinotoolkit/openvino) to Synet model conversion:

    /build/test_inference_engine -m convert -fm ie_model.xml -fw ie_weigths.bin -sm synet_model.xml -sw synet_weigths.bin

Also it is used in order to compare performance and accuracy of OpenVINO and Synet frameworks.
There are several test scripts:

* For manual testing you can use `./test.sh` (in the file you have to manually uncomment unit test that you need).
* Script `./check.sh` checks correctness of all tests.
* Script `./perf.sh` measures performance of Synet compare to OpenVINO.


ONNX test application
========================
The test application `test_onnx` is used for ONNX to Synet model conversion:

    /build/test_onnx -m convert -fw onnx_model.onnx -sm synet_model.xml -sw synet_weigths.bin

Also it is used in order to compare performance and accuracy of OpenVINO (it is used to infer ONNX models) and Synet frameworks.
There are several test scripts:

* For manual testing you can use `./test.sh` (in the file you have to manually uncomment unit test that you need).
* Script `./check.sh` checks correctness of all tests.
* Script `./perf.sh` measures performance of Synet compare to OpenVINO.

Precision test application
========================
The precision test application `test_precision` is used for independent accuracy testing of quantized Synet and OpenVINO models.
There is `./prec.sh` test script (in the file you have to manually uncomment unit test that you need).

Quantization test application
========================
The quantization test application `test_quantization` is used for INT8 quantization of FP32 Synet models and testing of them.
There is `./quant.sh` test script (in the file you have to manually uncomment unit test that you need).

Using samples
=======================================
The application `use_face_detection` is an example of using of Synet framework to face detection.

