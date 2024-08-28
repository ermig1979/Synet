Introduction
============

Synet is a small framework to infer neural network on CPU. Synet uses models previously trained by other deep neural network frameworks.

The main advantages  of Synet are:

* Synet is faster then most other DNN original frameworks (has great single thread CPU performance).
* Synet has next external dependencies - [Cpl](https://github.com/ermig1979/Cpl) and [Simd Library](https://github.com/ermig1979/Simd).

Building of test applications for Linux
==============================
To build test applications you can run following bash script:

    git clone -b master --recurse-submodules -v https://github.com/ermig1979/Synet.git clone
    cd clone
    ./build.sh

And applications `test_inference_engine`, `test_onnx`, `test_optimizer`, `test_precision`, `test_performance_difference`, 
`test_quantization`, `test_stability`, `use_face_detection` will be created in directory `build`.
There is a detail description of these test applications below.

OpenVINO test application
========================
The test application `test_inference_engine` is used for [OpenVINO](https://github.com/openvinotoolkit/openvino) to Synet model conversion:

    ./build/test_inference_engine -m=convert -fm=ie_model.xml -fw=ie_weigths.bin -sm=synet_model.xml -sw=synet_weigths.bin

Also it is used in order to compare performance and accuracy of OpenVINO and Synet frameworks.
The current Synet and OpenVINO frameworks support only Inference Engine models version 10. 
The previous versions of Inference Engine models are supported in [legacy_2020](https://github.com/ermig1979/Synet/tree/legacy_2020) branch.
There are several test scripts:

* For manual testing you can use `./test.sh` (in the file you have to manually uncomment unit test that you need).
* Script `./check.sh` checks correctness of all tests.
* Script `./perf.sh` measures performance of Synet compare to OpenVINO.


ONNX test application
========================
The test application `test_onnx` is used for ONNX to Synet model conversion:

    ./build/test_onnx -m=convert -fw=onnx_model.onnx -sm=synet_model.xml -sw=synet_weigths.bin

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

