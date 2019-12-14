Introduction
============

Synet is a small framework to inference neural network on CPU. Synet uses models previously trained by other deep neural network frameworks.

The main advantages  of Synet are:

* Synet is faster then most other DNN original frameworks (has great single thread CPU performance).
* Synet is header only, small C++ library.
* Synet has only one external dependence - [Simd Library](https://github.com/ermig1979/Simd).

Darknet Test project for Linux
==============================
To build test to compare Synet and [Darknet](https://github.com/pjreddie/darknet) for Linux you can run following bash script:

    ./build.sh darknet

And application `darknet_test` will be created in directory `build_darknet`.
In order to run this test use `./test.sh` bash script (in the file manually uncomment unit test that you need).

    ./test.sh 

Inference Engine Test project for Linux
=======================================
To build test to compare Synet and [Inference Engine](https://github.com/opencv/dldt) for Linux you can run following bash script:

    ./build.sh inference_engine

And application `inference_engine_test` will be created in directory `build_inference_engine`.
In order to run this test use `./test.sh` bash script (in the file manually uncomment unit test that you need).

Use samples for Linux
=======================================
To build use samples for Linux you can run following bash script:

    ./build.sh use_samples

And application `use_face_detection` will be created in directory `build_use_samples`.

Darknet model conversion
========================
In order to convert [Darknet](https://github.com/pjreddie/darknet) trained model to Synet model you can use `darknet_test` application:

	./build_darknet/darknet_test -m convert -om darknet_model.cfg -ow darknet_weigths.dat -sm synet_model.xml -sw synet_weigths.bin

Other model conversion
======================
In order to convert [Caffe](https://github.com/BVLC/caffe), [Tensorflow](https://github.com/tensorflow/tensorflow), [MXNet](https://mxnet.apache.org) or [ONNX](https://onnx.ai) trained models to Synet format you previously need to convert they to [Inference Engine](https://github.com/opencv/dldt) models format.
Then use `inference_engine_test` application:

	./build_inference_engine/inference_engine_test -m convert -om ie_model.xml -ow ie_weigths.bin -sm synet_model.xml -sw synet_weigths.bin




