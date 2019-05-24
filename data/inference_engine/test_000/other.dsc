<?xml version="1.0" ?>
<net batch="1" name="vehicle-attributes-recognition-barrier-0039" version="4">
	<layers>
		<layer id="0" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Convolution1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="37632"/>
				<biases offset="37656" size="256"/>
			</blobs>
		</layer>
		<layer id="3" name="ReLU1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Pooling1" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Convolution2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
			<blobs>
				<weights offset="37912" size="147456"/>
				<biases offset="185368" size="256"/>
			</blobs>
		</layer>
		<layer id="6" name="ReLU2" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Convolution3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
			<blobs>
				<weights offset="185624" size="147456"/>
			</blobs>
		</layer>
		<layer id="8" name="Eltwise1" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Mul_767/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
			<blobs>
				<weights offset="333080" size="256"/>
				<biases offset="333336" size="256"/>
			</blobs>
		</layer>
		<layer id="10" name="ReLU3" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="Convolution4" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
			<blobs>
				<weights offset="333592" size="294912"/>
				<biases offset="628504" size="512"/>
			</blobs>
		</layer>
		<layer id="12" name="ReLU4" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Convolution5" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
			<blobs>
				<weights offset="629016" size="589824"/>
			</blobs>
		</layer>
		<layer id="14" name="Convolution6" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1218840" size="32768"/>
			</blobs>
		</layer>
		<layer id="15" name="Eltwise2" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="Mul_770/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1251608" size="512"/>
				<biases offset="1252120" size="512"/>
			</blobs>
		</layer>
		<layer id="17" name="ReLU5" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="Convolution7_" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1252632" size="589824"/>
				<biases offset="1842456" size="512"/>
			</blobs>
		</layer>
		<layer id="19" name="ReLU6_" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="Convolution8_" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1842968" size="589824"/>
			</blobs>
		</layer>
		<layer id="21" name="Convolution9_" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2432792" size="65536"/>
			</blobs>
		</layer>
		<layer id="22" name="Eltwise3" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="conv_color" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="7" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>7</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2498328" size="3584"/>
				<biases offset="2501912" size="28"/>
			</blobs>
		</layer>
		<layer id="24" name="relu_conv_color" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>7</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>7</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="pool_color" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="5,5" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>7</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>7</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="color" precision="FP32" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>7</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>7</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="conv_type" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="4" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>4</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2501940" size="2048"/>
				<biases offset="2503988" size="16"/>
			</blobs>
		</layer>
		<layer id="28" name="relu_conv_type" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="pool_type" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="5,5" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="type" precision="FP32" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="8" to-port="1"/>
		<edge from-layer="8" from-port="2" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="3" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="13" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="13" from-port="2" to-layer="15" to-port="0"/>
		<edge from-layer="14" from-port="2" to-layer="15" to-port="1"/>
		<edge from-layer="15" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="3" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="20" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="21" to-port="0"/>
		<edge from-layer="20" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="2" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="26" to-port="0"/>
		<edge from-layer="22" from-port="2" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="3" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2019.1.0-178-ga427cda"/>
		<cli_parameters>
			<data_type value="FP32"/>
			<disable_fusing value="False"/>
			<disable_gfusing value="False"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<extensions value="DIR"/>
			<framework value="caffe"/>
			<generate_deprecated_IR_V2 value="False"/>
			<input value="input"/>
			<input_model value="DIR/resnet10_nolastlayer_iter_586000.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/test.prototxt"/>
			<input_shape value="[1,3,72,72]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'input': {'mean': None, 'scale': array([254.99991075])}}"/>
			<mean_values value="()"/>
			<model_name value="vehicle-attributes-recognition-barrier-0039"/>
			<move_to_preprocess value="False"/>
			<output value="['type', 'color']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'input': array([ 1,  3, 72, 72])}"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input[254.99991075003123]"/>
			<silent value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, finegrain_fusing, freeze_placeholder_with_value, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
