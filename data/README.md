----------------------
OpenVINO tests
======================

| Name | Description | Type | IE version | Xml | Bin |
| --- | --- | --- | --- | --- | --- |
| test_010f | Face detection | FP32 | 10 | [xml](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/face-detection-retail-0005/FP32/face-detection-retail-0005.xml) | [4.1 MB](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/face-detection-retail-0005/FP32/face-detection-retail-0005.bin) |
| test_011f | Vehicle detection | FP32 | 10 | [xml](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml) | [4.1 MB](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin) |
| test_012f | Person, vehicle, bike detection | FP32 | 10 | [xml](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.xml) | [6.9 MB](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.bin) |
| test_013f | Person, vehicle, bike detection | FP32 | 10 | [xml](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml) | [11.0 MB](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml) |
| test_014f | Licence plate recognition | FP32 | 10 | [xml](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.xml) | [4.6 MB](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.bin) |
| test_015f | Licence plate detection | FP32 | 10 | [xml](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml) | [2.6 MB](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin) |
| test_016f | Face attributes (age, gender) | FP32 | 10 | [xml](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml) | [8.2 MB](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin) |
| test_017f | Face detection | FP32 | 11 | [xml](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-0200/FP32/face-detection-0200.xml) | [6.9 MB](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-0200/FP32/face-detection-0200.bin) |
| test_018f | Vehicle attributes (type, color) | FP32 | 11 | [xml](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.xml) | [42.6 MB](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.bin) |
| test_019f | Face attributes (age, gender) | FP32 | 11 | [xml](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml) | [8.2 MB](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin) |

----------------------
ONNX tests
======================

| Name | Description | Architecture | Original |
| --- | --- | --- | --- |
| test_000 | Object classification | MobileNet-V2 | [13.6 MB](https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx) |
| test_001 | Face detection | UltraFace | [1.6 MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/models/onnx/version-RFB-640.onnx) |

----------------------
Quantization tests
======================

| Name | Description |
| --- | --- |
| test_003 | Face detection |
| test_010 | Face detection |

----------------------
Precision tests
======================

| Name | Description | Type |
| --- | --- | --- |
| test_003 | Face detection | FP32, INT8 |
| test_010 | Face detection | FP32, INT8 |
