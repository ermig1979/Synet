<?xml version="1.0" ?>
<net name="vehicle-license-plate-detection-barrier-0106" version="10">
	<layers>
		<layer id="0" name="Placeholder" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,300,300"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Placeholder/reverse_input_channels23530/Concat23549_const" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="1,3,1,1" size="12"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Mul_/Fused_Mul_" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Placeholder/reverse_input_channels23532/Concat23557_const" type="Const" version="opset1">
			<data element_type="f32" offset="12" shape="1,3,1,1" size="12"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Add_/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Placeholder/reverse_input_channels/Concat23541_const" type="Const" version="opset1">
			<data element_type="f32" offset="24" shape="16,3,3,3" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="MobilenetV2/Conv/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="data_add_1402314028/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1752" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="MobilenetV2/Conv/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="MobilenetV2/Conv/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="2432824331_const" type="Const" version="opset1">
			<data element_type="f32" offset="1816" shape="16,1,1,3,3" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="MobilenetV2/expanded_conv/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="data_add_1403114036/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2392" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="MobilenetV2/expanded_conv/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="MobilenetV2/expanded_conv/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="MobilenetV2/expanded_conv/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1513115133_const" type="Const" version="opset1">
			<data element_type="f32" offset="2456" shape="16,16,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="MobilenetV2/expanded_conv/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="data_add_1403914044/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3480" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="MobilenetV2/expanded_conv/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="MobilenetV2/expanded_conv/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="MobilenetV2/expanded_conv_1/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1513515137_const" type="Const" version="opset1">
			<data element_type="f32" offset="3544" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="MobilenetV2/expanded_conv_1/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="data_add_1404714052/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="9688" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="MobilenetV2/expanded_conv_1/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="MobilenetV2/expanded_conv_1/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="2431624319_const" type="Const" version="opset1">
			<data element_type="f32" offset="10072" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="MobilenetV2/expanded_conv_1/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="data_add_1405514060/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="13528" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="MobilenetV2/expanded_conv_1/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="MobilenetV2/expanded_conv_1/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="MobilenetV2/expanded_conv_1/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1514315145_const" type="Const" version="opset1">
			<data element_type="f32" offset="13912" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="MobilenetV2/expanded_conv_1/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="data_add_1406314068/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="20056" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="MobilenetV2/expanded_conv_1/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="MobilenetV2/expanded_conv_2/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1514715149_const" type="Const" version="opset1">
			<data element_type="f32" offset="20120" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="MobilenetV2/expanded_conv_2/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="data_add_1407114076/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="26264" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="MobilenetV2/expanded_conv_2/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="MobilenetV2/expanded_conv_2/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="2433624339_const" type="Const" version="opset1">
			<data element_type="f32" offset="26648" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="MobilenetV2/expanded_conv_2/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="data_add_1407914084/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="30104" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="MobilenetV2/expanded_conv_2/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="MobilenetV2/expanded_conv_2/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="MobilenetV2/expanded_conv_2/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1515515157_const" type="Const" version="opset1">
			<data element_type="f32" offset="30488" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="MobilenetV2/expanded_conv_2/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="data_add_1408714092/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="36632" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="MobilenetV2/expanded_conv_2/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="MobilenetV2/expanded_conv_2/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="MobilenetV2/expanded_conv_3/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1515915161_const" type="Const" version="opset1">
			<data element_type="f32" offset="36696" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="MobilenetV2/expanded_conv_3/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="data_add_1409514100/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="42840" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="MobilenetV2/expanded_conv_3/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="MobilenetV2/expanded_conv_3/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="2430424307_const" type="Const" version="opset1">
			<data element_type="f32" offset="43224" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="MobilenetV2/expanded_conv_3/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="data_add_1410314108/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="46680" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="MobilenetV2/expanded_conv_3/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="MobilenetV2/expanded_conv_3/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="MobilenetV2/expanded_conv_3/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1516715169_const" type="Const" version="opset1">
			<data element_type="f32" offset="47064" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="MobilenetV2/expanded_conv_3/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="data_add_1411114116/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="53208" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="MobilenetV2/expanded_conv_3/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="MobilenetV2/expanded_conv_4/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1517115173_const" type="Const" version="opset1">
			<data element_type="f32" offset="53272" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="MobilenetV2/expanded_conv_4/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="data_add_1411914124/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="59416" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="MobilenetV2/expanded_conv_4/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="MobilenetV2/expanded_conv_4/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="2434024343_const" type="Const" version="opset1">
			<data element_type="f32" offset="59800" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="MobilenetV2/expanded_conv_4/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="data_add_1412714132/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="63256" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="MobilenetV2/expanded_conv_4/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="MobilenetV2/expanded_conv_4/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="MobilenetV2/expanded_conv_4/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1517915181_const" type="Const" version="opset1">
			<data element_type="f32" offset="63640" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="MobilenetV2/expanded_conv_4/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="data_add_1413514140/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="69784" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="MobilenetV2/expanded_conv_4/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="MobilenetV2/expanded_conv_4/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="MobilenetV2/expanded_conv_5/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1518315185_const" type="Const" version="opset1">
			<data element_type="f32" offset="69848" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="MobilenetV2/expanded_conv_5/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="data_add_1414314148/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="75992" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="MobilenetV2/expanded_conv_5/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="MobilenetV2/expanded_conv_5/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="2435624359_const" type="Const" version="opset1">
			<data element_type="f32" offset="76376" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="MobilenetV2/expanded_conv_5/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="data_add_1415114156/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="79832" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="MobilenetV2/expanded_conv_5/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="MobilenetV2/expanded_conv_5/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1519115193_const" type="Const" version="opset1">
			<data element_type="f32" offset="80216" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="MobilenetV2/expanded_conv_5/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="data_add_1415914164/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="86360" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="MobilenetV2/expanded_conv_5/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="MobilenetV2/expanded_conv_6/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1519515197_const" type="Const" version="opset1">
			<data element_type="f32" offset="86424" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="MobilenetV2/expanded_conv_6/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="data_add_1416714172/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="92568" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="MobilenetV2/expanded_conv_6/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="MobilenetV2/expanded_conv_6/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="2434424347_const" type="Const" version="opset1">
			<data element_type="f32" offset="92952" shape="96,1,1,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="MobilenetV2/expanded_conv_6/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="data_add_1417514180/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="96408" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="MobilenetV2/expanded_conv_6/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="MobilenetV2/expanded_conv_6/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="MobilenetV2/expanded_conv_6/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1520315205_const" type="Const" version="opset1">
			<data element_type="f32" offset="96792" shape="24,96,1,1" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="MobilenetV2/expanded_conv_6/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="data_add_1418314188/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="106008" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="MobilenetV2/expanded_conv_6/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="MobilenetV2/expanded_conv_7/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1520715209_const" type="Const" version="opset1">
			<data element_type="f32" offset="106104" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="MobilenetV2/expanded_conv_7/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="data_add_1419114196/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="119928" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="MobilenetV2/expanded_conv_7/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="MobilenetV2/expanded_conv_7/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="2436424367_const" type="Const" version="opset1">
			<data element_type="f32" offset="120504" shape="144,1,1,3,3" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="MobilenetV2/expanded_conv_7/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="data_add_1419914204/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="125688" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="MobilenetV2/expanded_conv_7/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="MobilenetV2/expanded_conv_7/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="MobilenetV2/expanded_conv_7/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1521515217_const" type="Const" version="opset1">
			<data element_type="f32" offset="126264" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="MobilenetV2/expanded_conv_7/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="data_add_1420714212/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="140088" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="MobilenetV2/expanded_conv_7/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="MobilenetV2/expanded_conv_7/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="MobilenetV2/expanded_conv_8/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1521915221_const" type="Const" version="opset1">
			<data element_type="f32" offset="140184" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="MobilenetV2/expanded_conv_8/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="data_add_1421514220/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="154008" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="MobilenetV2/expanded_conv_8/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="MobilenetV2/expanded_conv_8/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="2434824351_const" type="Const" version="opset1">
			<data element_type="f32" offset="154584" shape="144,1,1,3,3" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="MobilenetV2/expanded_conv_8/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="data_add_1422314228/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="159768" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="MobilenetV2/expanded_conv_8/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="MobilenetV2/expanded_conv_8/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="MobilenetV2/expanded_conv_8/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1522715229_const" type="Const" version="opset1">
			<data element_type="f32" offset="160344" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="MobilenetV2/expanded_conv_8/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="data_add_1423114236/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="174168" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="MobilenetV2/expanded_conv_8/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="MobilenetV2/expanded_conv_8/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="MobilenetV2/expanded_conv_9/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1523115233_const" type="Const" version="opset1">
			<data element_type="f32" offset="174264" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="MobilenetV2/expanded_conv_9/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="data_add_1423914244/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="188088" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="MobilenetV2/expanded_conv_9/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="MobilenetV2/expanded_conv_9/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="2430024303_const" type="Const" version="opset1">
			<data element_type="f32" offset="188664" shape="144,1,1,3,3" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="MobilenetV2/expanded_conv_9/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="data_add_1424714252/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="193848" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="MobilenetV2/expanded_conv_9/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="MobilenetV2/expanded_conv_9/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="MobilenetV2/expanded_conv_9/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1523915241_const" type="Const" version="opset1">
			<data element_type="f32" offset="194424" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="MobilenetV2/expanded_conv_9/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="data_add_1425514260/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="208248" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="MobilenetV2/expanded_conv_9/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="MobilenetV2/expanded_conv_9/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="MobilenetV2/expanded_conv_10/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1524315245_const" type="Const" version="opset1">
			<data element_type="f32" offset="208344" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="MobilenetV2/expanded_conv_10/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="data_add_1426314268/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="222168" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="MobilenetV2/expanded_conv_10/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="MobilenetV2/expanded_conv_10/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="2436024363_const" type="Const" version="opset1">
			<data element_type="f32" offset="222744" shape="144,1,1,3,3" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="MobilenetV2/expanded_conv_10/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="159" name="data_add_1427114276/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="227928" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="160" name="MobilenetV2/expanded_conv_10/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="MobilenetV2/expanded_conv_10/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="162" name="MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1525115253_const" type="Const" version="opset1">
			<data element_type="f32" offset="228504" shape="32,144,1,1" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="MobilenetV2/expanded_conv_10/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="data_add_1427914284/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="246936" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="MobilenetV2/expanded_conv_11/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1525515257_const" type="Const" version="opset1">
			<data element_type="f32" offset="247064" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="MobilenetV2/expanded_conv_11/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="data_add_1428714292/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="271640" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="MobilenetV2/expanded_conv_11/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="MobilenetV2/expanded_conv_11/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="2435224355_const" type="Const" version="opset1">
			<data element_type="f32" offset="272408" shape="192,1,1,3,3" size="6912"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="MobilenetV2/expanded_conv_11/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="173" name="data_add_1429514300/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="279320" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="MobilenetV2/expanded_conv_11/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="MobilenetV2/expanded_conv_11/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="MobilenetV2/expanded_conv_11/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1526315265_const" type="Const" version="opset1">
			<data element_type="f32" offset="280088" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="177" name="MobilenetV2/expanded_conv_11/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="data_add_1430314308/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="304664" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="179" name="MobilenetV2/expanded_conv_11/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="MobilenetV2/expanded_conv_11/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="181" name="MobilenetV2/expanded_conv_12/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1526715269_const" type="Const" version="opset1">
			<data element_type="f32" offset="304792" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="MobilenetV2/expanded_conv_12/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="data_add_1431114316/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="329368" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="MobilenetV2/expanded_conv_12/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="MobilenetV2/expanded_conv_12/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="2430824311_const" type="Const" version="opset1">
			<data element_type="f32" offset="330136" shape="192,1,1,3,3" size="6912"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="MobilenetV2/expanded_conv_12/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="188" name="data_add_1431914324/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="337048" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="MobilenetV2/expanded_conv_12/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="MobilenetV2/expanded_conv_12/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="MobilenetV2/expanded_conv_12/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1527515277_const" type="Const" version="opset1">
			<data element_type="f32" offset="337816" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="192" name="MobilenetV2/expanded_conv_12/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="data_add_1432714332/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="362392" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="194" name="MobilenetV2/expanded_conv_12/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="MobilenetV2/expanded_conv_12/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="SSD/ssd_head/layer_14/output_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="362520" shape="8,32,3,3" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="SSD/ssd_head/layer_14/output_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="SSD/ssd_head/layer_14/output_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="371736" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="SSD/ssd_head/layer_14/output_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="SSD/ssd_head/layer_14/output_mbox_loc/BiasAdd/Add/Transpose/Cast_124765_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="SSD/ssd_head/layer_14/output_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="SSD/ssd_head/Flatten/flatten/Reshape/Cast_124749_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="203" name="SSD/ssd_head/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2888</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="MobilenetV2/expanded_conv_13/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1527915281_const" type="Const" version="opset1">
			<data element_type="f32" offset="371816" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="MobilenetV2/expanded_conv_13/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="data_add_1433514340/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="396392" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="207" name="MobilenetV2/expanded_conv_13/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="MobilenetV2/expanded_conv_13/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="2432424327_const" type="Const" version="opset1">
			<data element_type="f32" offset="397160" shape="192,1,1,3,3" size="6912"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="MobilenetV2/expanded_conv_13/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="data_add_1434314348/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="404072" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="212" name="MobilenetV2/expanded_conv_13/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="MobilenetV2/expanded_conv_13/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1528715289_const" type="Const" version="opset1">
			<data element_type="f32" offset="404840" shape="56,192,1,1" size="43008"/>
			<output>
				<port id="1" precision="FP32">
					<dim>56</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="215" name="MobilenetV2/expanded_conv_13/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>56</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="216" name="data_add_1435114356/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="447848" shape="1,56,1,1" size="224"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="MobilenetV2/expanded_conv_14/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1529115293_const" type="Const" version="opset1">
			<data element_type="f32" offset="448072" shape="336,56,1,1" size="75264"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="219" name="MobilenetV2/expanded_conv_14/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="data_add_1435914364/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="523336" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="221" name="MobilenetV2/expanded_conv_14/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="MobilenetV2/expanded_conv_14/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="2432024323_const" type="Const" version="opset1">
			<data element_type="f32" offset="524680" shape="336,1,1,3,3" size="12096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="MobilenetV2/expanded_conv_14/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="data_add_1436714372/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="536776" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="226" name="MobilenetV2/expanded_conv_14/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="MobilenetV2/expanded_conv_14/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="MobilenetV2/expanded_conv_14/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1529915301_const" type="Const" version="opset1">
			<data element_type="f32" offset="538120" shape="56,336,1,1" size="75264"/>
			<output>
				<port id="1" precision="FP32">
					<dim>56</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="MobilenetV2/expanded_conv_14/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>56</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="data_add_1437514380/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="613384" shape="1,56,1,1" size="224"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="MobilenetV2/expanded_conv_14/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="MobilenetV2/expanded_conv_14/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="MobilenetV2/expanded_conv_15/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1530315305_const" type="Const" version="opset1">
			<data element_type="f32" offset="613608" shape="336,56,1,1" size="75264"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="MobilenetV2/expanded_conv_15/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="data_add_1438314388/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="688872" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="MobilenetV2/expanded_conv_15/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="237" name="MobilenetV2/expanded_conv_15/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="238" name="2433224335_const" type="Const" version="opset1">
			<data element_type="f32" offset="690216" shape="336,1,1,3,3" size="12096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="MobilenetV2/expanded_conv_15/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="240" name="data_add_1439114396/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="702312" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="241" name="MobilenetV2/expanded_conv_15/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="MobilenetV2/expanded_conv_15/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="243" name="MobilenetV2/expanded_conv_15/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1531115313_const" type="Const" version="opset1">
			<data element_type="f32" offset="703656" shape="56,336,1,1" size="75264"/>
			<output>
				<port id="1" precision="FP32">
					<dim>56</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="MobilenetV2/expanded_conv_15/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>56</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="245" name="data_add_1439914404/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="778920" shape="1,56,1,1" size="224"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="246" name="MobilenetV2/expanded_conv_15/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="MobilenetV2/expanded_conv_15/add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="248" name="MobilenetV2/expanded_conv_16/expand/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1531515317_const" type="Const" version="opset1">
			<data element_type="f32" offset="779144" shape="336,56,1,1" size="75264"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="MobilenetV2/expanded_conv_16/expand/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="250" name="data_add_1440714412/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="854408" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="251" name="MobilenetV2/expanded_conv_16/expand/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="252" name="MobilenetV2/expanded_conv_16/expand/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="253" name="2431224315_const" type="Const" version="opset1">
			<data element_type="f32" offset="855752" shape="336,1,1,3,3" size="12096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="MobilenetV2/expanded_conv_16/depthwise/depthwise" type="GroupConvolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="255" name="data_add_1441514420/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="867848" shape="1,336,1,1" size="1344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="256" name="MobilenetV2/expanded_conv_16/depthwise/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="MobilenetV2/expanded_conv_16/depthwise/Relu6" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="258" name="MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm/mean/Fused_Mul_1532315325_const" type="Const" version="opset1">
			<data element_type="f32" offset="869192" shape="112,336,1,1" size="150528"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="MobilenetV2/expanded_conv_16/project/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>336</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>336</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="260" name="data_add_1442314428/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1019720" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="261" name="MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="SSD/ssd_head_1/layer_18/output_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1020168" shape="12,112,3,3" size="48384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="263" name="SSD/ssd_head_1/layer_18/output_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="264" name="SSD/ssd_head_1/layer_18/output_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1068552" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="265" name="SSD/ssd_head_1/layer_18/output_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="266" name="SSD/ssd_head_1/layer_18/output_mbox_loc/BiasAdd/Add/Transpose/Cast_124727_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="267" name="SSD/ssd_head_1/layer_18/output_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>10</dim>
					<dim>10</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="268" name="SSD/ssd_head_1/Flatten/flatten/Reshape/Cast_124777_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="SSD/ssd_head_1/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
					<dim>10</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1200</dim>
				</port>
			</output>
		</layer>
		<layer id="270" name="intermediate_1/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1068600" shape="96,112,1,1" size="43008"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="271" name="features/intermediate_1/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="intermediate_1/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1111608" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="273" name="features/intermediate_1/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="features/intermediate_1/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="275" name="feature_map_1/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1111992" shape="192,96,3,3" size="663552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="features/feature_map_1/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="feature_map_1/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1775544" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="features/feature_map_1/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="features/feature_map_1/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="SSD/ssd_head_2/feature_map_1_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1776312" shape="12,192,3,3" size="82944"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="SSD/ssd_head_2/feature_map_1_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="SSD/ssd_head_2/feature_map_1_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1859256" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="SSD/ssd_head_2/feature_map_1_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="SSD/ssd_head_2/feature_map_1_mbox_loc/BiasAdd/Add/Transpose/Cast_124733_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="SSD/ssd_head_2/feature_map_1_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5</dim>
					<dim>5</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="SSD/ssd_head_2/Flatten/flatten/Reshape/Cast_124725_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="SSD/ssd_head_2/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5</dim>
					<dim>5</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>300</dim>
				</port>
			</output>
		</layer>
		<layer id="288" name="intermediate_2/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1859304" shape="48,192,1,1" size="36864"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="289" name="features/intermediate_2/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="290" name="intermediate_2/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1896168" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="291" name="features/intermediate_2/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="292" name="features/intermediate_2/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="293" name="feature_map_2/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1896360" shape="96,48,3,3" size="165888"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="294" name="features/feature_map_2/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="295" name="feature_map_2/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2062248" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="features/feature_map_2/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="297" name="features/feature_map_2/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="298" name="SSD/ssd_head_3/feature_map_2_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2062632" shape="8,96,3,3" size="27648"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="299" name="SSD/ssd_head_3/feature_map_2_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="300" name="SSD/ssd_head_3/feature_map_2_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2090280" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="SSD/ssd_head_3/feature_map_2_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="SSD/ssd_head_3/feature_map_2_mbox_loc/BiasAdd/Add/Transpose/Cast_124763_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="SSD/ssd_head_3/feature_map_2_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="304" name="SSD/ssd_head_3/Flatten/flatten/Reshape/Cast_124723_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="305" name="SSD/ssd_head_3/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
				</port>
			</output>
		</layer>
		<layer id="306" name="intermediate_3/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2090312" shape="48,96,1,1" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="307" name="features/intermediate_3/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="308" name="intermediate_3/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2108744" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="309" name="features/intermediate_3/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="310" name="features/intermediate_3/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="311" name="feature_map_3/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2108936" shape="96,48,3,3" size="165888"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="312" name="features/feature_map_3/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="313" name="feature_map_3/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2274824" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="314" name="features/feature_map_3/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="315" name="features/feature_map_3/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="316" name="SSD/ssd_head_4/feature_map_3_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2275208" shape="8,96,3,3" size="27648"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="317" name="SSD/ssd_head_4/feature_map_3_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="318" name="SSD/ssd_head_4/feature_map_3_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2302856" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="319" name="SSD/ssd_head_4/feature_map_3_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="320" name="SSD/ssd_head_4/feature_map_3_mbox_loc/BiasAdd/Add/Transpose/Cast_124787_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="321" name="SSD/ssd_head_4/feature_map_3_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="322" name="SSD/ssd_head_4/Flatten/flatten/Reshape/Cast_124731_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="323" name="SSD/ssd_head_4/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="324" name="intermediate_4/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2302888" shape="24,96,1,1" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="325" name="features/intermediate_4/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="326" name="intermediate_4/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2312104" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="327" name="features/intermediate_4/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="328" name="features/intermediate_4/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="329" name="feature_map_4/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2312200" shape="48,24,3,3" size="41472"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="330" name="features/feature_map_4/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="331" name="feature_map_4/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2353672" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="332" name="features/feature_map_4/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="333" name="features/feature_map_4/Relu" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="334" name="SSD/ssd_head_5/feature_map_4_mbox_loc/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2353864" shape="12,48,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="335" name="SSD/ssd_head_5/feature_map_4_mbox_loc/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="336" name="SSD/ssd_head_5/feature_map_4_mbox_loc/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2374600" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="337" name="SSD/ssd_head_5/feature_map_4_mbox_loc/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="338" name="SSD/ssd_head_5/feature_map_4_mbox_loc/BiasAdd/Add/Transpose/Cast_124757_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="339" name="SSD/ssd_head_5/feature_map_4_mbox_loc/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="340" name="SSD/ssd_head_5/Flatten/flatten/Reshape/Cast_124781_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="341" name="SSD/ssd_head_5/Flatten/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="342" name="SSD/concat_reshape_softmax/mbox_loc" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2888</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1200</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>300</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>72</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>32</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="6" precision="FP32">
					<dim>1</dim>
					<dim>4504</dim>
				</port>
			</output>
		</layer>
		<layer id="343" name="SSD/concat_reshape_softmax/mbox_loc_final/Cast_124741_const" type="Const" version="opset1">
			<data element_type="i64" offset="2374648" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="344" name="SSD/concat_reshape_softmax/mbox_loc_final" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4504</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="345" name="DetectionOutput_Reshape_loc_/Cast_124773_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="346" name="DetectionOutput_Reshape_loc_" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4504</dim>
				</port>
			</output>
		</layer>
		<layer id="347" name="SSD/ssd_head/layer_14/output_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2374672" shape="6,32,3,3" size="6912"/>
			<output>
				<port id="1" precision="FP32">
					<dim>6</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="348" name="SSD/ssd_head/layer_14/output_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>6</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="349" name="SSD/ssd_head/layer_14/output_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2381584" shape="1,6,1,1" size="24"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="350" name="SSD/ssd_head/layer_14/output_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="351" name="SSD/ssd_head/layer_14/output_mbox_conf/BiasAdd/Add/Transpose/Cast_124745_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="352" name="SSD/ssd_head/layer_14/output_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="353" name="SSD/ssd_head/Flatten_1/flatten/Reshape/Cast_124759_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="354" name="SSD/ssd_head/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2166</dim>
				</port>
			</output>
		</layer>
		<layer id="355" name="SSD/ssd_head_1/layer_18/output_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2381608" shape="9,112,3,3" size="36288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>9</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="356" name="SSD/ssd_head_1/layer_18/output_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>9</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="357" name="SSD/ssd_head_1/layer_18/output_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2417896" shape="1,9,1,1" size="36"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="358" name="SSD/ssd_head_1/layer_18/output_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="359" name="SSD/ssd_head_1/layer_18/output_mbox_conf/BiasAdd/Add/Transpose/Cast_124739_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="360" name="SSD/ssd_head_1/layer_18/output_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>10</dim>
					<dim>10</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="361" name="SSD/ssd_head_1/Flatten_1/flatten/Reshape/Cast_124771_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="362" name="SSD/ssd_head_1/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>10</dim>
					<dim>10</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>900</dim>
				</port>
			</output>
		</layer>
		<layer id="363" name="SSD/ssd_head_2/feature_map_1_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2417932" shape="9,192,3,3" size="62208"/>
			<output>
				<port id="1" precision="FP32">
					<dim>9</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="364" name="SSD/ssd_head_2/feature_map_1_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>9</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="365" name="SSD/ssd_head_2/feature_map_1_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2480140" shape="1,9,1,1" size="36"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="366" name="SSD/ssd_head_2/feature_map_1_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="367" name="SSD/ssd_head_2/feature_map_1_mbox_conf/BiasAdd/Add/Transpose/Cast_124743_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="368" name="SSD/ssd_head_2/feature_map_1_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5</dim>
					<dim>5</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="369" name="SSD/ssd_head_2/Flatten_1/flatten/Reshape/Cast_124769_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="370" name="SSD/ssd_head_2/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5</dim>
					<dim>5</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>225</dim>
				</port>
			</output>
		</layer>
		<layer id="371" name="SSD/ssd_head_3/feature_map_2_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2480176" shape="6,96,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>6</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="372" name="SSD/ssd_head_3/feature_map_2_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>6</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="373" name="SSD/ssd_head_3/feature_map_2_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2500912" shape="1,6,1,1" size="24"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="374" name="SSD/ssd_head_3/feature_map_2_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="375" name="SSD/ssd_head_3/feature_map_2_mbox_conf/BiasAdd/Add/Transpose/Cast_124747_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="376" name="SSD/ssd_head_3/feature_map_2_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="377" name="SSD/ssd_head_3/Flatten_1/flatten/Reshape/Cast_124785_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="378" name="SSD/ssd_head_3/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>54</dim>
				</port>
			</output>
		</layer>
		<layer id="379" name="SSD/ssd_head_4/feature_map_3_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2500936" shape="6,96,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>6</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="380" name="SSD/ssd_head_4/feature_map_3_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>6</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="381" name="SSD/ssd_head_4/feature_map_3_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2521672" shape="1,6,1,1" size="24"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="382" name="SSD/ssd_head_4/feature_map_3_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="383" name="SSD/ssd_head_4/feature_map_3_mbox_conf/BiasAdd/Add/Transpose/Cast_124761_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="384" name="SSD/ssd_head_4/feature_map_3_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="385" name="SSD/ssd_head_4/Flatten_1/flatten/Reshape/Cast_124735_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="386" name="SSD/ssd_head_4/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="387" name="SSD/ssd_head_5/feature_map_4_mbox_conf/weights/read/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2521696" shape="9,48,3,3" size="15552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>9</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="388" name="SSD/ssd_head_5/feature_map_4_mbox_conf/Conv2D" type="Convolution" version="opset1">
			<data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>9</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="389" name="SSD/ssd_head_5/feature_map_4_mbox_conf/biases/read/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2537248" shape="1,9,1,1" size="36"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="390" name="SSD/ssd_head_5/feature_map_4_mbox_conf/BiasAdd/Add" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="391" name="SSD/ssd_head_5/feature_map_4_mbox_conf/BiasAdd/Add/Transpose/Cast_124729_const" type="Const" version="opset1">
			<data element_type="i64" offset="371768" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="392" name="SSD/ssd_head_5/feature_map_4_mbox_conf/BiasAdd/Add/Transpose" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>9</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="393" name="SSD/ssd_head_5/Flatten_1/flatten/Reshape/Cast_124767_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="394" name="SSD/ssd_head_5/Flatten_1/flatten/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>9</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="395" name="SSD/concat_reshape_softmax/mbox_conf" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2166</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>900</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>225</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>54</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>24</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="6" precision="FP32">
					<dim>1</dim>
					<dim>3378</dim>
				</port>
			</output>
		</layer>
		<layer id="396" name="SSD/concat_reshape_softmax/mbox_conf_logits/Cast_124737_const" type="Const" version="opset1">
			<data element_type="i64" offset="2537284" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="397" name="SSD/concat_reshape_softmax/mbox_conf_logits" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3378</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="398" name="SSD/concat_reshape_softmax/concat/values_0/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i32" offset="2537308" shape="1" size="4"/>
			<output>
				<port id="1" precision="I32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="399" name="SSD/concat_reshape_softmax/Shape_1" type="ShapeOf" version="opset3">
			<data output_type="i32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="400" name="SSD/concat_reshape_softmax/Slice/Cast_124751_const" type="Const" version="opset1">
			<data element_type="i64" offset="2537312" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="401" name="SSD/concat_reshape_softmax/Slice/Cast_224753_const" type="Const" version="opset1">
			<data element_type="i64" offset="2537320" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="402" name="SSD/concat_reshape_softmax/Slice/Cast_324755_const" type="Const" version="opset1">
			<data element_type="i64" offset="2537328" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="403" name="SSD/concat_reshape_softmax/Slice" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="404" name="SSD/concat_reshape_softmax/concat" type="Concat" version="opset1">
			<data axis="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I32">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="405" name="SSD/concat_reshape_softmax/Reshape/Cast_1" type="Convert" version="opset1">
			<data destination_type="i64"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="406" name="SSD/concat_reshape_softmax/Reshape" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="407" name="SSD/concat_reshape_softmax/Softmax" type="SoftMax" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="408" name="SSD/concat_reshape_softmax/Shape" type="ShapeOf" version="opset3">
			<data output_type="i32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="409" name="SSD/concat_reshape_softmax/mbox_conf_final/Cast_1" type="Convert" version="opset1">
			<data destination_type="i64"/>
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="410" name="SSD/concat_reshape_softmax/mbox_conf_final" type="Reshape" version="opset1">
			<data special_zero="False"/>
			<input>
				<port id="0">
					<dim>1126</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="411" name="DetectionOutput_Reshape_conf_/Cast_124775_const" type="Const" version="opset1">
			<data element_type="i64" offset="371800" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="412" name="DetectionOutput_Reshape_conf_" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1126</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3378</dim>
				</port>
			</output>
		</layer>
		<layer id="413" name="DetectionOutput_Reshape_priors_/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2537336" shape="1,2,4504" size="36032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>4504</dim>
				</port>
			</output>
		</layer>
		<layer id="414" name="DetectionOutput_" type="DetectionOutput" version="opset1">
			<data code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.01" input_height="1" input_width="1" keep_top_k="200" nms_threshold="0.45" normalized="1" num_classes="3" pad_mode="caffe.ResizeParameter.CONSTANT" resize_mode="caffe.ResizeParameter.WARP" share_location="1" top_k="400" variance_encoded_in_target="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4504</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3378</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>4504</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>200</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="415" name="sink_" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>200</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="6" from-port="2" to-layer="8" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="1"/>
		<edge from-layer="8" from-port="2" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="11" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="11" to-port="1"/>
		<edge from-layer="11" from-port="2" to-layer="13" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="13" to-port="1"/>
		<edge from-layer="13" from-port="2" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="1"/>
		<edge from-layer="16" from-port="2" to-layer="18" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="1"/>
		<edge from-layer="18" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="19" to-port="1"/>
		<edge from-layer="19" from-port="2" to-layer="21" to-port="0"/>
		<edge from-layer="20" from-port="1" to-layer="21" to-port="1"/>
		<edge from-layer="21" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="1"/>
		<edge from-layer="23" from-port="2" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="26" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="26" to-port="1"/>
		<edge from-layer="26" from-port="2" to-layer="28" to-port="0"/>
		<edge from-layer="27" from-port="1" to-layer="28" to-port="1"/>
		<edge from-layer="28" from-port="2" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="31" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="31" to-port="1"/>
		<edge from-layer="31" from-port="2" to-layer="33" to-port="0"/>
		<edge from-layer="32" from-port="1" to-layer="33" to-port="1"/>
		<edge from-layer="33" from-port="2" to-layer="35" to-port="0"/>
		<edge from-layer="34" from-port="1" to-layer="35" to-port="1"/>
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="1" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="40" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="40" to-port="1"/>
		<edge from-layer="40" from-port="2" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="45" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="45" to-port="1"/>
		<edge from-layer="45" from-port="2" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="1" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="33" from-port="2" to-layer="48" to-port="1"/>
		<edge from-layer="48" from-port="2" to-layer="50" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="50" to-port="1"/>
		<edge from-layer="50" from-port="2" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="1" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="55" to-port="0"/>
		<edge from-layer="54" from-port="1" to-layer="55" to-port="1"/>
		<edge from-layer="55" from-port="2" to-layer="57" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="57" to-port="1"/>
		<edge from-layer="57" from-port="2" to-layer="58" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="60" to-port="0"/>
		<edge from-layer="59" from-port="1" to-layer="60" to-port="1"/>
		<edge from-layer="60" from-port="2" to-layer="62" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="62" to-port="1"/>
		<edge from-layer="62" from-port="2" to-layer="64" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="1"/>
		<edge from-layer="64" from-port="2" to-layer="66" to-port="0"/>
		<edge from-layer="65" from-port="1" to-layer="66" to-port="1"/>
		<edge from-layer="66" from-port="2" to-layer="67" to-port="0"/>
		<edge from-layer="67" from-port="1" to-layer="69" to-port="0"/>
		<edge from-layer="68" from-port="1" to-layer="69" to-port="1"/>
		<edge from-layer="69" from-port="2" to-layer="71" to-port="0"/>
		<edge from-layer="70" from-port="1" to-layer="71" to-port="1"/>
		<edge from-layer="71" from-port="2" to-layer="72" to-port="0"/>
		<edge from-layer="72" from-port="1" to-layer="74" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="74" to-port="1"/>
		<edge from-layer="74" from-port="2" to-layer="76" to-port="0"/>
		<edge from-layer="75" from-port="1" to-layer="76" to-port="1"/>
		<edge from-layer="76" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="62" from-port="2" to-layer="77" to-port="1"/>
		<edge from-layer="77" from-port="2" to-layer="79" to-port="0"/>
		<edge from-layer="78" from-port="1" to-layer="79" to-port="1"/>
		<edge from-layer="79" from-port="2" to-layer="81" to-port="0"/>
		<edge from-layer="80" from-port="1" to-layer="81" to-port="1"/>
		<edge from-layer="81" from-port="2" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="1" to-layer="84" to-port="0"/>
		<edge from-layer="83" from-port="1" to-layer="84" to-port="1"/>
		<edge from-layer="84" from-port="2" to-layer="86" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="86" to-port="1"/>
		<edge from-layer="86" from-port="2" to-layer="87" to-port="0"/>
		<edge from-layer="87" from-port="1" to-layer="89" to-port="0"/>
		<edge from-layer="88" from-port="1" to-layer="89" to-port="1"/>
		<edge from-layer="89" from-port="2" to-layer="91" to-port="0"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="1"/>
		<edge from-layer="91" from-port="2" to-layer="92" to-port="0"/>
		<edge from-layer="77" from-port="2" to-layer="92" to-port="1"/>
		<edge from-layer="92" from-port="2" to-layer="94" to-port="0"/>
		<edge from-layer="93" from-port="1" to-layer="94" to-port="1"/>
		<edge from-layer="94" from-port="2" to-layer="96" to-port="0"/>
		<edge from-layer="95" from-port="1" to-layer="96" to-port="1"/>
		<edge from-layer="96" from-port="2" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="99" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="99" to-port="1"/>
		<edge from-layer="99" from-port="2" to-layer="101" to-port="0"/>
		<edge from-layer="100" from-port="1" to-layer="101" to-port="1"/>
		<edge from-layer="101" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="1" to-layer="104" to-port="0"/>
		<edge from-layer="103" from-port="1" to-layer="104" to-port="1"/>
		<edge from-layer="104" from-port="2" to-layer="106" to-port="0"/>
		<edge from-layer="105" from-port="1" to-layer="106" to-port="1"/>
		<edge from-layer="106" from-port="2" to-layer="108" to-port="0"/>
		<edge from-layer="107" from-port="1" to-layer="108" to-port="1"/>
		<edge from-layer="108" from-port="2" to-layer="110" to-port="0"/>
		<edge from-layer="109" from-port="1" to-layer="110" to-port="1"/>
		<edge from-layer="110" from-port="2" to-layer="111" to-port="0"/>
		<edge from-layer="111" from-port="1" to-layer="113" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="113" to-port="1"/>
		<edge from-layer="113" from-port="2" to-layer="115" to-port="0"/>
		<edge from-layer="114" from-port="1" to-layer="115" to-port="1"/>
		<edge from-layer="115" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="116" from-port="1" to-layer="118" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="118" to-port="1"/>
		<edge from-layer="118" from-port="2" to-layer="120" to-port="0"/>
		<edge from-layer="119" from-port="1" to-layer="120" to-port="1"/>
		<edge from-layer="120" from-port="2" to-layer="121" to-port="0"/>
		<edge from-layer="106" from-port="2" to-layer="121" to-port="1"/>
		<edge from-layer="121" from-port="2" to-layer="123" to-port="0"/>
		<edge from-layer="122" from-port="1" to-layer="123" to-port="1"/>
		<edge from-layer="123" from-port="2" to-layer="125" to-port="0"/>
		<edge from-layer="124" from-port="1" to-layer="125" to-port="1"/>
		<edge from-layer="125" from-port="2" to-layer="126" to-port="0"/>
		<edge from-layer="126" from-port="1" to-layer="128" to-port="0"/>
		<edge from-layer="127" from-port="1" to-layer="128" to-port="1"/>
		<edge from-layer="128" from-port="2" to-layer="130" to-port="0"/>
		<edge from-layer="129" from-port="1" to-layer="130" to-port="1"/>
		<edge from-layer="130" from-port="2" to-layer="131" to-port="0"/>
		<edge from-layer="131" from-port="1" to-layer="133" to-port="0"/>
		<edge from-layer="132" from-port="1" to-layer="133" to-port="1"/>
		<edge from-layer="133" from-port="2" to-layer="135" to-port="0"/>
		<edge from-layer="134" from-port="1" to-layer="135" to-port="1"/>
		<edge from-layer="135" from-port="2" to-layer="136" to-port="0"/>
		<edge from-layer="121" from-port="2" to-layer="136" to-port="1"/>
		<edge from-layer="136" from-port="2" to-layer="138" to-port="0"/>
		<edge from-layer="137" from-port="1" to-layer="138" to-port="1"/>
		<edge from-layer="138" from-port="2" to-layer="140" to-port="0"/>
		<edge from-layer="139" from-port="1" to-layer="140" to-port="1"/>
		<edge from-layer="140" from-port="2" to-layer="141" to-port="0"/>
		<edge from-layer="141" from-port="1" to-layer="143" to-port="0"/>
		<edge from-layer="142" from-port="1" to-layer="143" to-port="1"/>
		<edge from-layer="143" from-port="2" to-layer="145" to-port="0"/>
		<edge from-layer="144" from-port="1" to-layer="145" to-port="1"/>
		<edge from-layer="145" from-port="2" to-layer="146" to-port="0"/>
		<edge from-layer="146" from-port="1" to-layer="148" to-port="0"/>
		<edge from-layer="147" from-port="1" to-layer="148" to-port="1"/>
		<edge from-layer="148" from-port="2" to-layer="150" to-port="0"/>
		<edge from-layer="149" from-port="1" to-layer="150" to-port="1"/>
		<edge from-layer="150" from-port="2" to-layer="151" to-port="0"/>
		<edge from-layer="136" from-port="2" to-layer="151" to-port="1"/>
		<edge from-layer="151" from-port="2" to-layer="153" to-port="0"/>
		<edge from-layer="152" from-port="1" to-layer="153" to-port="1"/>
		<edge from-layer="153" from-port="2" to-layer="155" to-port="0"/>
		<edge from-layer="154" from-port="1" to-layer="155" to-port="1"/>
		<edge from-layer="155" from-port="2" to-layer="156" to-port="0"/>
		<edge from-layer="156" from-port="1" to-layer="158" to-port="0"/>
		<edge from-layer="157" from-port="1" to-layer="158" to-port="1"/>
		<edge from-layer="158" from-port="2" to-layer="160" to-port="0"/>
		<edge from-layer="159" from-port="1" to-layer="160" to-port="1"/>
		<edge from-layer="160" from-port="2" to-layer="161" to-port="0"/>
		<edge from-layer="161" from-port="1" to-layer="163" to-port="0"/>
		<edge from-layer="162" from-port="1" to-layer="163" to-port="1"/>
		<edge from-layer="163" from-port="2" to-layer="165" to-port="0"/>
		<edge from-layer="164" from-port="1" to-layer="165" to-port="1"/>
		<edge from-layer="165" from-port="2" to-layer="167" to-port="0"/>
		<edge from-layer="166" from-port="1" to-layer="167" to-port="1"/>
		<edge from-layer="167" from-port="2" to-layer="169" to-port="0"/>
		<edge from-layer="168" from-port="1" to-layer="169" to-port="1"/>
		<edge from-layer="169" from-port="2" to-layer="170" to-port="0"/>
		<edge from-layer="170" from-port="1" to-layer="172" to-port="0"/>
		<edge from-layer="171" from-port="1" to-layer="172" to-port="1"/>
		<edge from-layer="172" from-port="2" to-layer="174" to-port="0"/>
		<edge from-layer="173" from-port="1" to-layer="174" to-port="1"/>
		<edge from-layer="174" from-port="2" to-layer="175" to-port="0"/>
		<edge from-layer="175" from-port="1" to-layer="177" to-port="0"/>
		<edge from-layer="176" from-port="1" to-layer="177" to-port="1"/>
		<edge from-layer="177" from-port="2" to-layer="179" to-port="0"/>
		<edge from-layer="178" from-port="1" to-layer="179" to-port="1"/>
		<edge from-layer="179" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="165" from-port="2" to-layer="180" to-port="1"/>
		<edge from-layer="180" from-port="2" to-layer="182" to-port="0"/>
		<edge from-layer="181" from-port="1" to-layer="182" to-port="1"/>
		<edge from-layer="182" from-port="2" to-layer="184" to-port="0"/>
		<edge from-layer="183" from-port="1" to-layer="184" to-port="1"/>
		<edge from-layer="184" from-port="2" to-layer="185" to-port="0"/>
		<edge from-layer="185" from-port="1" to-layer="187" to-port="0"/>
		<edge from-layer="186" from-port="1" to-layer="187" to-port="1"/>
		<edge from-layer="187" from-port="2" to-layer="189" to-port="0"/>
		<edge from-layer="188" from-port="1" to-layer="189" to-port="1"/>
		<edge from-layer="189" from-port="2" to-layer="190" to-port="0"/>
		<edge from-layer="190" from-port="1" to-layer="192" to-port="0"/>
		<edge from-layer="191" from-port="1" to-layer="192" to-port="1"/>
		<edge from-layer="192" from-port="2" to-layer="194" to-port="0"/>
		<edge from-layer="193" from-port="1" to-layer="194" to-port="1"/>
		<edge from-layer="194" from-port="2" to-layer="195" to-port="0"/>
		<edge from-layer="180" from-port="2" to-layer="195" to-port="1"/>
		<edge from-layer="195" from-port="2" to-layer="197" to-port="0"/>
		<edge from-layer="196" from-port="1" to-layer="197" to-port="1"/>
		<edge from-layer="197" from-port="2" to-layer="199" to-port="0"/>
		<edge from-layer="198" from-port="1" to-layer="199" to-port="1"/>
		<edge from-layer="199" from-port="2" to-layer="201" to-port="0"/>
		<edge from-layer="200" from-port="1" to-layer="201" to-port="1"/>
		<edge from-layer="201" from-port="2" to-layer="203" to-port="0"/>
		<edge from-layer="202" from-port="1" to-layer="203" to-port="1"/>
		<edge from-layer="195" from-port="2" to-layer="205" to-port="0"/>
		<edge from-layer="204" from-port="1" to-layer="205" to-port="1"/>
		<edge from-layer="205" from-port="2" to-layer="207" to-port="0"/>
		<edge from-layer="206" from-port="1" to-layer="207" to-port="1"/>
		<edge from-layer="207" from-port="2" to-layer="208" to-port="0"/>
		<edge from-layer="208" from-port="1" to-layer="210" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="210" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="212" to-port="0"/>
		<edge from-layer="211" from-port="1" to-layer="212" to-port="1"/>
		<edge from-layer="212" from-port="2" to-layer="213" to-port="0"/>
		<edge from-layer="213" from-port="1" to-layer="215" to-port="0"/>
		<edge from-layer="214" from-port="1" to-layer="215" to-port="1"/>
		<edge from-layer="215" from-port="2" to-layer="217" to-port="0"/>
		<edge from-layer="216" from-port="1" to-layer="217" to-port="1"/>
		<edge from-layer="217" from-port="2" to-layer="219" to-port="0"/>
		<edge from-layer="218" from-port="1" to-layer="219" to-port="1"/>
		<edge from-layer="219" from-port="2" to-layer="221" to-port="0"/>
		<edge from-layer="220" from-port="1" to-layer="221" to-port="1"/>
		<edge from-layer="221" from-port="2" to-layer="222" to-port="0"/>
		<edge from-layer="222" from-port="1" to-layer="224" to-port="0"/>
		<edge from-layer="223" from-port="1" to-layer="224" to-port="1"/>
		<edge from-layer="224" from-port="2" to-layer="226" to-port="0"/>
		<edge from-layer="225" from-port="1" to-layer="226" to-port="1"/>
		<edge from-layer="226" from-port="2" to-layer="227" to-port="0"/>
		<edge from-layer="227" from-port="1" to-layer="229" to-port="0"/>
		<edge from-layer="228" from-port="1" to-layer="229" to-port="1"/>
		<edge from-layer="229" from-port="2" to-layer="231" to-port="0"/>
		<edge from-layer="230" from-port="1" to-layer="231" to-port="1"/>
		<edge from-layer="231" from-port="2" to-layer="232" to-port="0"/>
		<edge from-layer="217" from-port="2" to-layer="232" to-port="1"/>
		<edge from-layer="232" from-port="2" to-layer="234" to-port="0"/>
		<edge from-layer="233" from-port="1" to-layer="234" to-port="1"/>
		<edge from-layer="234" from-port="2" to-layer="236" to-port="0"/>
		<edge from-layer="235" from-port="1" to-layer="236" to-port="1"/>
		<edge from-layer="236" from-port="2" to-layer="237" to-port="0"/>
		<edge from-layer="237" from-port="1" to-layer="239" to-port="0"/>
		<edge from-layer="238" from-port="1" to-layer="239" to-port="1"/>
		<edge from-layer="239" from-port="2" to-layer="241" to-port="0"/>
		<edge from-layer="240" from-port="1" to-layer="241" to-port="1"/>
		<edge from-layer="241" from-port="2" to-layer="242" to-port="0"/>
		<edge from-layer="242" from-port="1" to-layer="244" to-port="0"/>
		<edge from-layer="243" from-port="1" to-layer="244" to-port="1"/>
		<edge from-layer="244" from-port="2" to-layer="246" to-port="0"/>
		<edge from-layer="245" from-port="1" to-layer="246" to-port="1"/>
		<edge from-layer="246" from-port="2" to-layer="247" to-port="0"/>
		<edge from-layer="232" from-port="2" to-layer="247" to-port="1"/>
		<edge from-layer="247" from-port="2" to-layer="249" to-port="0"/>
		<edge from-layer="248" from-port="1" to-layer="249" to-port="1"/>
		<edge from-layer="249" from-port="2" to-layer="251" to-port="0"/>
		<edge from-layer="250" from-port="1" to-layer="251" to-port="1"/>
		<edge from-layer="251" from-port="2" to-layer="252" to-port="0"/>
		<edge from-layer="252" from-port="1" to-layer="254" to-port="0"/>
		<edge from-layer="253" from-port="1" to-layer="254" to-port="1"/>
		<edge from-layer="254" from-port="2" to-layer="256" to-port="0"/>
		<edge from-layer="255" from-port="1" to-layer="256" to-port="1"/>
		<edge from-layer="256" from-port="2" to-layer="257" to-port="0"/>
		<edge from-layer="257" from-port="1" to-layer="259" to-port="0"/>
		<edge from-layer="258" from-port="1" to-layer="259" to-port="1"/>
		<edge from-layer="259" from-port="2" to-layer="261" to-port="0"/>
		<edge from-layer="260" from-port="1" to-layer="261" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="263" to-port="0"/>
		<edge from-layer="262" from-port="1" to-layer="263" to-port="1"/>
		<edge from-layer="263" from-port="2" to-layer="265" to-port="0"/>
		<edge from-layer="264" from-port="1" to-layer="265" to-port="1"/>
		<edge from-layer="265" from-port="2" to-layer="267" to-port="0"/>
		<edge from-layer="266" from-port="1" to-layer="267" to-port="1"/>
		<edge from-layer="267" from-port="2" to-layer="269" to-port="0"/>
		<edge from-layer="268" from-port="1" to-layer="269" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="271" to-port="0"/>
		<edge from-layer="270" from-port="1" to-layer="271" to-port="1"/>
		<edge from-layer="271" from-port="2" to-layer="273" to-port="0"/>
		<edge from-layer="272" from-port="1" to-layer="273" to-port="1"/>
		<edge from-layer="273" from-port="2" to-layer="274" to-port="0"/>
		<edge from-layer="274" from-port="1" to-layer="276" to-port="0"/>
		<edge from-layer="275" from-port="1" to-layer="276" to-port="1"/>
		<edge from-layer="276" from-port="2" to-layer="278" to-port="0"/>
		<edge from-layer="277" from-port="1" to-layer="278" to-port="1"/>
		<edge from-layer="278" from-port="2" to-layer="279" to-port="0"/>
		<edge from-layer="279" from-port="1" to-layer="281" to-port="0"/>
		<edge from-layer="280" from-port="1" to-layer="281" to-port="1"/>
		<edge from-layer="281" from-port="2" to-layer="283" to-port="0"/>
		<edge from-layer="282" from-port="1" to-layer="283" to-port="1"/>
		<edge from-layer="283" from-port="2" to-layer="285" to-port="0"/>
		<edge from-layer="284" from-port="1" to-layer="285" to-port="1"/>
		<edge from-layer="285" from-port="2" to-layer="287" to-port="0"/>
		<edge from-layer="286" from-port="1" to-layer="287" to-port="1"/>
		<edge from-layer="279" from-port="1" to-layer="289" to-port="0"/>
		<edge from-layer="288" from-port="1" to-layer="289" to-port="1"/>
		<edge from-layer="289" from-port="2" to-layer="291" to-port="0"/>
		<edge from-layer="290" from-port="1" to-layer="291" to-port="1"/>
		<edge from-layer="291" from-port="2" to-layer="292" to-port="0"/>
		<edge from-layer="292" from-port="1" to-layer="294" to-port="0"/>
		<edge from-layer="293" from-port="1" to-layer="294" to-port="1"/>
		<edge from-layer="294" from-port="2" to-layer="296" to-port="0"/>
		<edge from-layer="295" from-port="1" to-layer="296" to-port="1"/>
		<edge from-layer="296" from-port="2" to-layer="297" to-port="0"/>
		<edge from-layer="297" from-port="1" to-layer="299" to-port="0"/>
		<edge from-layer="298" from-port="1" to-layer="299" to-port="1"/>
		<edge from-layer="299" from-port="2" to-layer="301" to-port="0"/>
		<edge from-layer="300" from-port="1" to-layer="301" to-port="1"/>
		<edge from-layer="301" from-port="2" to-layer="303" to-port="0"/>
		<edge from-layer="302" from-port="1" to-layer="303" to-port="1"/>
		<edge from-layer="303" from-port="2" to-layer="305" to-port="0"/>
		<edge from-layer="304" from-port="1" to-layer="305" to-port="1"/>
		<edge from-layer="297" from-port="1" to-layer="307" to-port="0"/>
		<edge from-layer="306" from-port="1" to-layer="307" to-port="1"/>
		<edge from-layer="307" from-port="2" to-layer="309" to-port="0"/>
		<edge from-layer="308" from-port="1" to-layer="309" to-port="1"/>
		<edge from-layer="309" from-port="2" to-layer="310" to-port="0"/>
		<edge from-layer="310" from-port="1" to-layer="312" to-port="0"/>
		<edge from-layer="311" from-port="1" to-layer="312" to-port="1"/>
		<edge from-layer="312" from-port="2" to-layer="314" to-port="0"/>
		<edge from-layer="313" from-port="1" to-layer="314" to-port="1"/>
		<edge from-layer="314" from-port="2" to-layer="315" to-port="0"/>
		<edge from-layer="315" from-port="1" to-layer="317" to-port="0"/>
		<edge from-layer="316" from-port="1" to-layer="317" to-port="1"/>
		<edge from-layer="317" from-port="2" to-layer="319" to-port="0"/>
		<edge from-layer="318" from-port="1" to-layer="319" to-port="1"/>
		<edge from-layer="319" from-port="2" to-layer="321" to-port="0"/>
		<edge from-layer="320" from-port="1" to-layer="321" to-port="1"/>
		<edge from-layer="321" from-port="2" to-layer="323" to-port="0"/>
		<edge from-layer="322" from-port="1" to-layer="323" to-port="1"/>
		<edge from-layer="315" from-port="1" to-layer="325" to-port="0"/>
		<edge from-layer="324" from-port="1" to-layer="325" to-port="1"/>
		<edge from-layer="325" from-port="2" to-layer="327" to-port="0"/>
		<edge from-layer="326" from-port="1" to-layer="327" to-port="1"/>
		<edge from-layer="327" from-port="2" to-layer="328" to-port="0"/>
		<edge from-layer="328" from-port="1" to-layer="330" to-port="0"/>
		<edge from-layer="329" from-port="1" to-layer="330" to-port="1"/>
		<edge from-layer="330" from-port="2" to-layer="332" to-port="0"/>
		<edge from-layer="331" from-port="1" to-layer="332" to-port="1"/>
		<edge from-layer="332" from-port="2" to-layer="333" to-port="0"/>
		<edge from-layer="333" from-port="1" to-layer="335" to-port="0"/>
		<edge from-layer="334" from-port="1" to-layer="335" to-port="1"/>
		<edge from-layer="335" from-port="2" to-layer="337" to-port="0"/>
		<edge from-layer="336" from-port="1" to-layer="337" to-port="1"/>
		<edge from-layer="337" from-port="2" to-layer="339" to-port="0"/>
		<edge from-layer="338" from-port="1" to-layer="339" to-port="1"/>
		<edge from-layer="339" from-port="2" to-layer="341" to-port="0"/>
		<edge from-layer="340" from-port="1" to-layer="341" to-port="1"/>
		<edge from-layer="203" from-port="2" to-layer="342" to-port="0"/>
		<edge from-layer="269" from-port="2" to-layer="342" to-port="1"/>
		<edge from-layer="287" from-port="2" to-layer="342" to-port="2"/>
		<edge from-layer="305" from-port="2" to-layer="342" to-port="3"/>
		<edge from-layer="323" from-port="2" to-layer="342" to-port="4"/>
		<edge from-layer="341" from-port="2" to-layer="342" to-port="5"/>
		<edge from-layer="342" from-port="6" to-layer="344" to-port="0"/>
		<edge from-layer="343" from-port="1" to-layer="344" to-port="1"/>
		<edge from-layer="344" from-port="2" to-layer="346" to-port="0"/>
		<edge from-layer="345" from-port="1" to-layer="346" to-port="1"/>
		<edge from-layer="195" from-port="2" to-layer="348" to-port="0"/>
		<edge from-layer="347" from-port="1" to-layer="348" to-port="1"/>
		<edge from-layer="348" from-port="2" to-layer="350" to-port="0"/>
		<edge from-layer="349" from-port="1" to-layer="350" to-port="1"/>
		<edge from-layer="350" from-port="2" to-layer="352" to-port="0"/>
		<edge from-layer="351" from-port="1" to-layer="352" to-port="1"/>
		<edge from-layer="352" from-port="2" to-layer="354" to-port="0"/>
		<edge from-layer="353" from-port="1" to-layer="354" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="356" to-port="0"/>
		<edge from-layer="355" from-port="1" to-layer="356" to-port="1"/>
		<edge from-layer="356" from-port="2" to-layer="358" to-port="0"/>
		<edge from-layer="357" from-port="1" to-layer="358" to-port="1"/>
		<edge from-layer="358" from-port="2" to-layer="360" to-port="0"/>
		<edge from-layer="359" from-port="1" to-layer="360" to-port="1"/>
		<edge from-layer="360" from-port="2" to-layer="362" to-port="0"/>
		<edge from-layer="361" from-port="1" to-layer="362" to-port="1"/>
		<edge from-layer="279" from-port="1" to-layer="364" to-port="0"/>
		<edge from-layer="363" from-port="1" to-layer="364" to-port="1"/>
		<edge from-layer="364" from-port="2" to-layer="366" to-port="0"/>
		<edge from-layer="365" from-port="1" to-layer="366" to-port="1"/>
		<edge from-layer="366" from-port="2" to-layer="368" to-port="0"/>
		<edge from-layer="367" from-port="1" to-layer="368" to-port="1"/>
		<edge from-layer="368" from-port="2" to-layer="370" to-port="0"/>
		<edge from-layer="369" from-port="1" to-layer="370" to-port="1"/>
		<edge from-layer="297" from-port="1" to-layer="372" to-port="0"/>
		<edge from-layer="371" from-port="1" to-layer="372" to-port="1"/>
		<edge from-layer="372" from-port="2" to-layer="374" to-port="0"/>
		<edge from-layer="373" from-port="1" to-layer="374" to-port="1"/>
		<edge from-layer="374" from-port="2" to-layer="376" to-port="0"/>
		<edge from-layer="375" from-port="1" to-layer="376" to-port="1"/>
		<edge from-layer="376" from-port="2" to-layer="378" to-port="0"/>
		<edge from-layer="377" from-port="1" to-layer="378" to-port="1"/>
		<edge from-layer="315" from-port="1" to-layer="380" to-port="0"/>
		<edge from-layer="379" from-port="1" to-layer="380" to-port="1"/>
		<edge from-layer="380" from-port="2" to-layer="382" to-port="0"/>
		<edge from-layer="381" from-port="1" to-layer="382" to-port="1"/>
		<edge from-layer="382" from-port="2" to-layer="384" to-port="0"/>
		<edge from-layer="383" from-port="1" to-layer="384" to-port="1"/>
		<edge from-layer="384" from-port="2" to-layer="386" to-port="0"/>
		<edge from-layer="385" from-port="1" to-layer="386" to-port="1"/>
		<edge from-layer="333" from-port="1" to-layer="388" to-port="0"/>
		<edge from-layer="387" from-port="1" to-layer="388" to-port="1"/>
		<edge from-layer="388" from-port="2" to-layer="390" to-port="0"/>
		<edge from-layer="389" from-port="1" to-layer="390" to-port="1"/>
		<edge from-layer="390" from-port="2" to-layer="392" to-port="0"/>
		<edge from-layer="391" from-port="1" to-layer="392" to-port="1"/>
		<edge from-layer="392" from-port="2" to-layer="394" to-port="0"/>
		<edge from-layer="393" from-port="1" to-layer="394" to-port="1"/>
		<edge from-layer="354" from-port="2" to-layer="395" to-port="0"/>
		<edge from-layer="362" from-port="2" to-layer="395" to-port="1"/>
		<edge from-layer="370" from-port="2" to-layer="395" to-port="2"/>
		<edge from-layer="378" from-port="2" to-layer="395" to-port="3"/>
		<edge from-layer="386" from-port="2" to-layer="395" to-port="4"/>
		<edge from-layer="394" from-port="2" to-layer="395" to-port="5"/>
		<edge from-layer="395" from-port="6" to-layer="397" to-port="0"/>
		<edge from-layer="396" from-port="1" to-layer="397" to-port="1"/>
		<edge from-layer="397" from-port="2" to-layer="399" to-port="0"/>
		<edge from-layer="399" from-port="1" to-layer="403" to-port="0"/>
		<edge from-layer="400" from-port="1" to-layer="403" to-port="1"/>
		<edge from-layer="401" from-port="1" to-layer="403" to-port="2"/>
		<edge from-layer="402" from-port="1" to-layer="403" to-port="3"/>
		<edge from-layer="398" from-port="1" to-layer="404" to-port="0"/>
		<edge from-layer="403" from-port="4" to-layer="404" to-port="1"/>
		<edge from-layer="404" from-port="2" to-layer="405" to-port="0"/>
		<edge from-layer="397" from-port="2" to-layer="406" to-port="0"/>
		<edge from-layer="405" from-port="1" to-layer="406" to-port="1"/>
		<edge from-layer="406" from-port="2" to-layer="407" to-port="0"/>
		<edge from-layer="397" from-port="2" to-layer="408" to-port="0"/>
		<edge from-layer="408" from-port="1" to-layer="409" to-port="0"/>
		<edge from-layer="407" from-port="1" to-layer="410" to-port="0"/>
		<edge from-layer="409" from-port="1" to-layer="410" to-port="1"/>
		<edge from-layer="410" from-port="2" to-layer="412" to-port="0"/>
		<edge from-layer="411" from-port="1" to-layer="412" to-port="1"/>
		<edge from-layer="346" from-port="2" to-layer="414" to-port="0"/>
		<edge from-layer="412" from-port="2" to-layer="414" to-port="1"/>
		<edge from-layer="413" from-port="1" to-layer="414" to-port="2"/>
		<edge from-layer="414" from-port="3" to-layer="415" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2021.1.0-1223-31b3e356abc-releases/2021/1"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP32"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<disable_weights_compression value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="tf"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input value="Placeholder"/>
			<input_model value="DIR/MobileNetSSD.pb.frozen"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,300,300,3]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'Placeholder': {'mean': array([127.5, 127.5, 127.5]), 'scale': array([127.5])}}"/>
			<mean_values value="Placeholder[127.5,127.5,127.5]"/>
			<model_name value="vehicle-license-plate-detection-barrier-0106"/>
			<output value="['SSD/concat_reshape_softmax/mbox_loc_final', 'SSD/concat_reshape_softmax/mbox_conf_final', 'SSD/concat_reshape_softmax/mbox_priorbox']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'Placeholder': array([  1, 300, 300,   3])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="Placeholder[127.5]"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<transformations_config value="DIR/MobileNetSSD.tfmo.json"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
