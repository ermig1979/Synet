<?xml version="1.0" ?>
<net batch="1" name="age_gender" version="4">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="5184"/>
				<biases offset="5184" size="192"/>
			</blobs>
		</layer>
		<layer id="2" name="pool1" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="relu1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5376" size="110592"/>
				<biases offset="115968" size="256"/>
			</blobs>
		</layer>
		<layer id="5" name="pool2" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="relu2" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="conv3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="96" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
			<blobs>
				<weights offset="116224" size="221184"/>
				<biases offset="337408" size="384"/>
			</blobs>
		</layer>
		<layer id="8" name="pool3" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="relu3" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv4" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="337792" size="663552"/>
				<biases offset="1001344" size="768"/>
			</blobs>
		</layer>
		<layer id="11" name="relu4" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="conv5" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1002112" size="1769472"/>
				<biases offset="2771584" size="1024"/>
			</blobs>
		</layer>
		<layer id="13" name="relu5" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="age_conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2772608" size="2359296"/>
				<biases offset="5131904" size="1024"/>
			</blobs>
		</layer>
		<layer id="15" name="relu6_a" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="age_conv2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="512" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5132928" size="524288"/>
				<biases offset="5657216" size="2048"/>
			</blobs>
		</layer>
		<layer id="17" name="relu7_a" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="age_conv3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="1" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5659264" size="2048"/>
				<biases offset="5661312" size="4"/>
			</blobs>
		</layer>
		<layer id="19" name="gender_conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5661316" size="2359296"/>
				<biases offset="8020612" size="1024"/>
			</blobs>
		</layer>
		<layer id="20" name="relu6_g" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="gender_conv2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="512" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8021636" size="524288"/>
				<biases offset="8545924" size="2048"/>
			</blobs>
		</layer>
		<layer id="22" name="relu7_g" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="gender_conv3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="2" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8547972" size="4096"/>
				<biases offset="8552068" size="8"/>
			</blobs>
		</layer>
		<layer id="24" name="prob" precision="FP32" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="3" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="3" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="3" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="1" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
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
			<input value="data"/>
			<input_model value="DIR/age_gender_net.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/age_gender_net.prototxt"/>
			<input_shape value="[1,3,62,62]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'scale': array([254.99991075]), 'mean': None}}"/>
			<mean_values value="()"/>
			<model_name value="age-gender-recognition-retail-0013"/>
			<move_to_preprocess value="False"/>
			<output value="['prob', 'age_conv3']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'data': array([ 1,  3, 62, 62])}"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="data[254.99991075003123]"/>
			<silent value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, finegrain_fusing, freeze_placeholder_with_value, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
