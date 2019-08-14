<?xml version="1.0" ?>
<net batch="1" name="landmarks-regression-retail-0009" version="4">
	<layers>
		<layer id="0" name="0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="68" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="16" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="1728"/>
				<biases offset="1752" size="64"/>
			</blobs>
		</layer>
		<layer id="3" name="69" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1816" size="4"/>
			</blobs>
		</layer>
		<layer id="4" name="70" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>48</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Mul1_1007/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1820" size="64"/>
				<biases offset="1884" size="64"/>
			</blobs>
		</layer>
		<layer id="6" name="72" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="32" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1948" size="18432"/>
				<biases offset="20380" size="128"/>
			</blobs>
		</layer>
		<layer id="7" name="73" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</output>
			<blobs>
				<weights offset="20508" size="4"/>
			</blobs>
		</layer>
		<layer id="8" name="74" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>24</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Mul1_962/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="20512" size="128"/>
				<biases offset="20640" size="128"/>
			</blobs>
		</layer>
		<layer id="10" name="76" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="20768" size="73728"/>
				<biases offset="94496" size="256"/>
			</blobs>
		</layer>
		<layer id="11" name="77" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</output>
			<blobs>
				<weights offset="94752" size="4"/>
			</blobs>
		</layer>
		<layer id="12" name="78" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>12</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Mul1_989/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="94756" size="256"/>
				<biases offset="95012" size="256"/>
			</blobs>
		</layer>
		<layer id="14" name="80" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="95268" size="147456"/>
				<biases offset="242724" size="256"/>
			</blobs>
		</layer>
		<layer id="15" name="81" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="242980" size="4"/>
			</blobs>
		</layer>
		<layer id="16" name="Mul1_980/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="242984" size="256"/>
				<biases offset="243240" size="256"/>
			</blobs>
		</layer>
		<layer id="17" name="83" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="243496" size="294912"/>
				<biases offset="538408" size="512"/>
			</blobs>
		</layer>
		<layer id="18" name="84" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="538920" size="4"/>
			</blobs>
		</layer>
		<layer id="19" name="Mul1_/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="538924" size="512"/>
				<biases offset="539436" size="512"/>
			</blobs>
		</layer>
		<layer id="20" name="86" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="6,6" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>6</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="539948" size="18432"/>
				<biases offset="558380" size="512"/>
			</blobs>
		</layer>
		<layer id="21" name="87" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="558892" size="4"/>
			</blobs>
		</layer>
		<layer id="22" name="Mul1_1016/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="558896" size="512"/>
				<biases offset="559408" size="512"/>
			</blobs>
		</layer>
		<layer id="23" name="89" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
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
				<weights offset="559920" size="131072"/>
				<biases offset="690992" size="1024"/>
			</blobs>
		</layer>
		<layer id="24" name="90" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="692016" size="4"/>
			</blobs>
		</layer>
		<layer id="25" name="Mul1_971/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
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
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="692020" size="1024"/>
				<biases offset="693044" size="1024"/>
			</blobs>
		</layer>
		<layer id="26" name="92" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="694068" size="65536"/>
				<biases offset="759604" size="256"/>
			</blobs>
		</layer>
		<layer id="27" name="93" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="759860" size="4"/>
			</blobs>
		</layer>
		<layer id="28" name="94" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="10" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>10</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="759864" size="2560"/>
				<biases offset="762424" size="40"/>
			</blobs>
		</layer>
		<layer id="29" name="95" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>10</dim>
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
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="3" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="2" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="3" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="3" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="3" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="3" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="3" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="2" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="3" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="3" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="2" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="3" to-layer="29" to-port="0"/>
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
			<framework value="onnx"/>
			<generate_deprecated_IR_V2 value="False"/>
			<input value="0"/>
			<input_model value="DIR/Landnet_wing_blur2_21000.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,48,48]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'0': {'mean': None, 'scale': array([254.99991075])}}"/>
			<mean_values value="()"/>
			<model_name value="landmarks-regression-retail-0009"/>
			<move_to_preprocess value="False"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'0': array([ 1,  3, 48, 48])}"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="0[254.99991075003123]"/>
			<silent value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, finegrain_fusing, freeze_placeholder_with_value, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, output, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
