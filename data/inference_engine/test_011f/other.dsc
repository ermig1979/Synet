<?xml version="1.0" ?>
<net name="vd-net-caffe" version="10">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,384,672"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data_mul_1051510519/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="1,1,1,1" size="4"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
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
					<dim>384</dim>
					<dim>672</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="data_add_1051610521/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4" shape="1,3,1,1" size="12"/>
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
					<dim>384</dim>
					<dim>672</dim>
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
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Mul1_9339/Fused_Mul_1130211304_const" type="Const" version="opset1">
			<data element_type="f32" offset="16" shape="32,3,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="conv1" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="data_add_1052410529/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3472" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="Add1_9340/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
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
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="relu1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="1719117194_const" type="Const" version="opset1">
			<data element_type="f32" offset="3600" shape="32,1,1,3,3" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="conv2_1/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="data_add_1053210537/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4752" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Add1_9484/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
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
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="relu2_1/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="Mul1_/Fused_Mul_1131011312_const" type="Const" version="opset1">
			<data element_type="f32" offset="4880" shape="56,32,1,1" size="7168"/>
			<output>
				<port id="1" precision="FP32">
					<dim>56</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="conv2_1/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
				<port id="1">
					<dim>56</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="data_add_1054010545/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="12048" shape="1,56,1,1" size="224"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="Add1_/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
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
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="relu2_1/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="1716717170_const" type="Const" version="opset1">
			<data element_type="f32" offset="12272" shape="56,1,1,3,3" size="2016"/>
			<output>
				<port id="1" precision="FP32">
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="conv2_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
				<port id="1">
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="data_add_1054810553/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="14288" shape="1,56,1,1" size="224"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="Add1_9316/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
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
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="relu2_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="Mul1_9543/Fused_Mul_1131811320_const" type="Const" version="opset1">
			<data element_type="f32" offset="14512" shape="112,56,1,1" size="25088"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="conv2_2/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>56</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="data_add_1055610561/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="39600" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="Add1_9544/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
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
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="relu2_2/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="1715517158_const" type="Const" version="opset1">
			<data element_type="f32" offset="40048" shape="112,1,1,3,3" size="4032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="conv3_1/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="data_add_1056410569/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="44080" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="Add1_9256/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
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
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="relu3_1/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="Mul1_9603/Fused_Mul_1132611328_const" type="Const" version="opset1">
			<data element_type="f32" offset="44528" shape="112,112,1,1" size="50176"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="conv3_1/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="data_add_1057210577/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="94704" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="Add1_9604/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
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
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="relu3_1/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="1720717210_const" type="Const" version="opset1">
			<data element_type="f32" offset="95152" shape="112,1,1,3,3" size="4032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="conv3_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="data_add_1058010585/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="99184" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="Add1_9400/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>48</dim>
					<dim>84</dim>
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
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="relu3_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="Mul1_9663/Fused_Mul_1133411336_const" type="Const" version="opset1">
			<data element_type="f32" offset="99632" shape="208,112,1,1" size="93184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>208</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="conv3_2/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>208</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="data_add_1058810593/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="192816" shape="1,208,1,1" size="832"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="Add1_9664/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="relu3_2/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="1718317186_const" type="Const" version="opset1">
			<data element_type="f32" offset="193648" shape="208,1,1,3,3" size="7488"/>
			<output>
				<port id="1" precision="FP32">
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="conv4_1/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="data_add_1059610601/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="201136" shape="1,208,1,1" size="832"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="Add1_9388/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="relu4_1/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="Mul1_9567/Fused_Mul_1134211344_const" type="Const" version="opset1">
			<data element_type="f32" offset="201968" shape="216,208,1,1" size="179712"/>
			<output>
				<port id="1" precision="FP32">
					<dim>216</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="conv4_1/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>216</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="data_add_1060410609/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="381680" shape="1,216,1,1" size="864"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="Add1_9568/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="relu4_1/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="1719917202_const" type="Const" version="opset1">
			<data element_type="f32" offset="382544" shape="216,1,1,3,3" size="7776"/>
			<output>
				<port id="1" precision="FP32">
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="conv4_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
				<port id="1">
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="data_add_1061210617/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="390320" shape="1,216,1,1" size="864"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="Add1_9628/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="relu4_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="Mul1_9351/Fused_Mul_1135011352_const" type="Const" version="opset1">
			<data element_type="f32" offset="391184" shape="328,216,1,1" size="283392"/>
			<output>
				<port id="1" precision="FP32">
					<dim>328</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="conv4_2/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>328</dim>
					<dim>216</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="data_add_1062010625/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="674576" shape="1,328,1,1" size="1312"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="Add1_9352/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="relu4_2/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="1717117174_const" type="Const" version="opset1">
			<data element_type="f32" offset="675888" shape="328,1,1,3,3" size="11808"/>
			<output>
				<port id="1" precision="FP32">
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="conv5_1/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="data_add_1062810633/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="687696" shape="1,328,1,1" size="1312"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="Add1_9460/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="relu5_1/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="Mul1_9279/Fused_Mul_1135811360_const" type="Const" version="opset1">
			<data element_type="f32" offset="689008" shape="288,328,1,1" size="377856"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="conv5_1/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>328</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>328</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="data_add_1063610641/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1066864" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="Add1_9280/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="relu5_1/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="1714317146_const" type="Const" version="opset1">
			<data element_type="f32" offset="1068016" shape="288,1,1,3,3" size="10368"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="conv5_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="data_add_1064410649/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1078384" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="Add1_9580/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="relu5_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="Mul1_9471/Fused_Mul_1136611368_const" type="Const" version="opset1">
			<data element_type="f32" offset="1079536" shape="288,288,1,1" size="331776"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="conv5_2/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="data_add_1065210657/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1411312" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="Add1_9472/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="relu5_2/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="1716317166_const" type="Const" version="opset1">
			<data element_type="f32" offset="1412464" shape="288,1,1,3,3" size="10368"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="conv5_3/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="data_add_1066010665/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1422832" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="Add1_9496/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="relu5_3/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="Mul1_9651/Fused_Mul_1137411376_const" type="Const" version="opset1">
			<data element_type="f32" offset="1423984" shape="240,288,1,1" size="276480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>240</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="conv5_3/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>240</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="data_add_1066810673/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1700464" shape="1,240,1,1" size="960"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="Add1_9652/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="relu5_3/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="1717517178_const" type="Const" version="opset1">
			<data element_type="f32" offset="1701424" shape="240,1,1,3,3" size="8640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="conv5_4/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="data_add_1067610681/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1710064" shape="1,240,1,1" size="960"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="Add1_9640/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="relu5_4/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="Mul1_9519/Fused_Mul_1138211384_const" type="Const" version="opset1">
			<data element_type="f32" offset="1711024" shape="264,240,1,1" size="253440"/>
			<output>
				<port id="1" precision="FP32">
					<dim>264</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="conv5_4/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>264</dim>
					<dim>240</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="data_add_1068410689/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1964464" shape="1,264,1,1" size="1056"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="Add1_9520/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="relu5_4/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="1715117154_const" type="Const" version="opset1">
			<data element_type="f32" offset="1965520" shape="264,1,1,3,3" size="9504"/>
			<output>
				<port id="1" precision="FP32">
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="conv5_5/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="data_add_1069210697/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1975024" shape="1,264,1,1" size="1056"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="Add1_9232/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="relu5_5/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="Mul1_9423/Fused_Mul_1139011392_const" type="Const" version="opset1">
			<data element_type="f32" offset="1976080" shape="192,264,1,1" size="202752"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="conv5_5/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>264</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>192</dim>
					<dim>264</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="data_add_1070010705/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2178832" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="Add1_9424/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="relu5_5/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="311/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2179600" shape="16,192,3,3" size="110592"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="conv4_3_0_norm_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="conv4_3_0_norm_mbox_loc/Dims6452/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2290192" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="conv4_3_0_norm_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="conv4_3_0_norm_mbox_loc_perm/Cast_117626_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="conv4_3_0_norm_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="conv4_3_0_norm_mbox_loc_flat/Cast_117586_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="conv4_3_0_norm_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="406/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2290304" shape="16,192,3,3" size="110592"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="conv4_3_norm_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="conv4_3_norm_mbox_loc/Dims6416/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2400896" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="conv4_3_norm_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="conv4_3_norm_mbox_loc_perm/Cast_117682_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="conv4_3_norm_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="conv4_3_norm_mbox_loc_flat/Cast_117660_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="conv4_3_norm_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="1720317206_const" type="Const" version="opset1">
			<data element_type="f32" offset="2400960" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="137" name="conv5_6/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="data_add_1070810713/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2407872" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="Add1_9376/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="relu5_6/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="Mul1_9243/Fused_Mul_1139811400_const" type="Const" version="opset1">
			<data element_type="f32" offset="2408640" shape="208,192,1,1" size="159744"/>
			<output>
				<port id="1" precision="FP32">
					<dim>208</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="conv5_6/sep/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>208</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="data_add_1071610721/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2568384" shape="1,208,1,1" size="832"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="conv5_6/sep/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="relu5_6/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="1714717150_const" type="Const" version="opset1">
			<data element_type="f32" offset="2569216" shape="208,1,1,3,3" size="7488"/>
			<output>
				<port id="1" precision="FP32">
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="conv6/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="data_add_1072410729/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2576704" shape="1,208,1,1" size="832"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="Add1_9304/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="relu6/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="Mul1_9219/Fused_Mul_1140611408_const" type="Const" version="opset1">
			<data element_type="f32" offset="2577536" shape="88,208,1,1" size="73216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>88</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="conv6/sep" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>88</dim>
					<dim>208</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="data_add_1073210737/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2650752" shape="1,88,1,1" size="352"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="Add1_9220/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="relu6/sep" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="256/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2651104" shape="24,88,3,3" size="76032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="fc7_0_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="fc7_0_mbox_loc/Dims6410/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2727136" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="159" name="fc7_0_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="160" name="fc7_0_mbox_loc_perm/Cast_117714_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="fc7_0_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="162" name="fc7_0_mbox_loc_flat/Cast_117680_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="fc7_0_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="507/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2727232" shape="24,88,3,3" size="76032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="fc7_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="fc7_mbox_loc/Dims6470/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2803264" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="fc7_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="fc7_mbox_loc_perm/Cast_117698_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="fc7_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="fc7_mbox_loc_flat/Cast_117686_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="fc7_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="Mul1_9435/Fused_Mul_1141011412_const" type="Const" version="opset1">
			<data element_type="f32" offset="2803360" shape="96,88,1,1" size="33792"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="173" name="conv6_1" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="data_add_1074010745/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2837152" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="Add1_9436/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="relu6_1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="177" name="1718717190_const" type="Const" version="opset1">
			<data element_type="f32" offset="2837536" shape="96,1,1,3,3" size="3456"/>
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
		<layer id="178" name="conv6_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="179" name="data_add_1074810753/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2840992" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="Add1_9268/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
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
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="181" name="relu6_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="Mul1_9507/Fused_Mul_1141811420_const" type="Const" version="opset1">
			<data element_type="f32" offset="2841376" shape="152,96,1,1" size="58368"/>
			<output>
				<port id="1" precision="FP32">
					<dim>152</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="conv6_2" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>152</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="data_add_1075610761/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2899744" shape="1,152,1,1" size="608"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>152</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="Add1_9508/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>152</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="relu6_2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="366/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2900352" shape="24,152,3,3" size="131328"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>152</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="188" name="conv6_2_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>152</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="conv6_2_mbox_loc/Dims6440/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3031680" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="conv6_2_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
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
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="conv6_2_mbox_loc_perm/Cast_117594_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="192" name="conv6_2_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="conv6_2_mbox_loc_flat/Cast_117570_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="194" name="conv6_2_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="Mul1_9327/Fused_Mul_1142211424_const" type="Const" version="opset1">
			<data element_type="f32" offset="3031776" shape="112,152,1,1" size="68096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>152</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="conv7_1" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>152</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="data_add_1076410769/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3099872" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="Add1_9328/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>6</dim>
					<dim>11</dim>
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
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="relu7_1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="1717917182_const" type="Const" version="opset1">
			<data element_type="f32" offset="3100320" shape="112,1,1,3,3" size="4032"/>
			<output>
				<port id="1" precision="FP32">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="conv7_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="data_add_1077210777/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3104352" shape="1,112,1,1" size="448"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="203" name="Add1_9616/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>6</dim>
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
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="relu7_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="Mul1_9363/Fused_Mul_1143011432_const" type="Const" version="opset1">
			<data element_type="f32" offset="3104800" shape="200,112,1,1" size="89600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>200</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="conv7_2" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>200</dim>
					<dim>112</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="207" name="data_add_1078010785/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3194400" shape="1,200,1,1" size="800"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>200</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="Add1_9364/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>200</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="relu7_2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="373/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3195200" shape="24,200,3,3" size="172800"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="conv7_2_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="212" name="conv7_2_mbox_loc/Dims6404/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3368000" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="conv7_2_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
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
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="conv7_2_mbox_loc_perm/Cast_117684_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="215" name="conv7_2_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="216" name="conv7_2_mbox_loc_flat/Cast_117582_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="conv7_2_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="Mul1_9531/Fused_Mul_1143411436_const" type="Const" version="opset1">
			<data element_type="f32" offset="3368096" shape="120,200,1,1" size="96000"/>
			<output>
				<port id="1" precision="FP32">
					<dim>120</dim>
					<dim>200</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="219" name="conv8_1" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>120</dim>
					<dim>200</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="data_add_1078810793/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3464096" shape="1,120,1,1" size="480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="221" name="Add1_9532/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="relu8_1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="1715917162_const" type="Const" version="opset1">
			<data element_type="f32" offset="3464576" shape="120,1,1,3,3" size="4320"/>
			<output>
				<port id="1" precision="FP32">
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="conv8_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="data_add_1079610801/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3468896" shape="1,120,1,1" size="480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="226" name="Add1_9448/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="relu8_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="Mul1_9411/Fused_Mul_1144211444_const" type="Const" version="opset1">
			<data element_type="f32" offset="3469376" shape="224,120,1,1" size="107520"/>
			<output>
				<port id="1" precision="FP32">
					<dim>224</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="conv8_2" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>224</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="data_add_1080410809/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3576896" shape="1,224,1,1" size="896"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="Add1_9412/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>224</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="relu8_2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="370/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3577792" shape="16,224,3,3" size="129024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>224</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="conv8_2_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>224</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="conv8_2_mbox_loc/Dims6464/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3706816" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="conv8_2_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>2</dim>
					<dim>3</dim>
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
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="237" name="conv8_2_mbox_loc_perm/Cast_117584_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="238" name="conv8_2_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="conv8_2_mbox_loc_flat/Cast_117616_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="240" name="conv8_2_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="241" name="Mul1_9591/Fused_Mul_1144611448_const" type="Const" version="opset1">
			<data element_type="f32" offset="3706880" shape="64,224,1,1" size="57344"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>224</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="conv9_1" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>224</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="243" name="data_add_1081210817/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3764224" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="Add1_9592/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="245" name="relu9_1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="246" name="1719517198_const" type="Const" version="opset1">
			<data element_type="f32" offset="3764480" shape="64,1,1,3,3" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="conv9_2/dw" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="248" name="data_add_1082010825/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3766784" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="Add1_9556/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="250" name="relu9_2/dw" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="251" name="Mul1_9291/Fused_Mul_1145411456_const" type="Const" version="opset1">
			<data element_type="f32" offset="3767040" shape="128,64,1,1" size="32768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="252" name="conv9_2" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>128</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="253" name="data_add_1082810833/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3799808" shape="1,128,1,1" size="512"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="Add1_9292/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="255" name="relu9_2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="256" name="344/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3800320" shape="16,128,3,3" size="73728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="conv9_2_mbox_loc/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="258" name="conv9_2_mbox_loc/Dims6392/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3874048" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="conv9_2_mbox_loc" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>2</dim>
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
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="260" name="conv9_2_mbox_loc_perm/Cast_117630_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="261" name="conv9_2_mbox_loc_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="conv9_2_mbox_loc_flat/Cast_117638_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="263" name="conv9_2_mbox_loc_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>16</dim>
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
		<layer id="264" name="mbox_loc" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1584</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>432</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>96</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="8" precision="FP32">
					<dim>1</dim>
					<dim>46496</dim>
				</port>
			</output>
		</layer>
		<layer id="265" name="476/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3874112" shape="8,192,3,3" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="266" name="conv4_3_0_norm_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="267" name="conv4_3_0_norm_mbox_conf/Dims6446/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3929408" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="268" name="conv4_3_0_norm_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="conv4_3_0_norm_mbox_conf_perm/Cast_117688_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="270" name="conv4_3_0_norm_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="271" name="conv4_3_0_norm_mbox_conf_flat/Cast_117628_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="conv4_3_0_norm_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
			</output>
		</layer>
		<layer id="273" name="339/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3929440" shape="8,192,3,3" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="conv4_3_norm_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="275" name="conv4_3_norm_mbox_conf/Dims6386/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3984736" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="conv4_3_norm_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
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
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="conv4_3_norm_mbox_conf_perm/Cast_117602_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="conv4_3_norm_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="conv4_3_norm_mbox_conf_flat/Cast_117724_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="conv4_3_norm_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="402/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3984768" shape="12,88,3,3" size="38016"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="fc7_0_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="fc7_0_mbox_conf/Dims6422/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4022784" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="fc7_0_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="fc7_0_mbox_conf_perm/Cast_117704_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="fc7_0_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="fc7_0_mbox_conf_flat/Cast_117574_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="288" name="fc7_0_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
			</output>
		</layer>
		<layer id="289" name="368/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4022832" shape="12,88,3,3" size="38016"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="290" name="fc7_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>88</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="291" name="fc7_mbox_conf/Dims6458/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4060848" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="292" name="fc7_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
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
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="293" name="fc7_mbox_conf_perm/Cast_117572_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="294" name="fc7_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="295" name="fc7_mbox_conf_flat/Cast_117702_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="fc7_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
			</output>
		</layer>
		<layer id="297" name="283/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4060896" shape="12,152,3,3" size="65664"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>152</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="298" name="conv6_2_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>152</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="299" name="conv6_2_mbox_conf/Dims6428/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4126560" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="300" name="conv6_2_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
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
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="conv6_2_mbox_conf_perm/Cast_117696_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="conv6_2_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="conv6_2_mbox_conf_flat/Cast_117700_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="304" name="conv6_2_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>792</dim>
				</port>
			</output>
		</layer>
		<layer id="305" name="302/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4126608" shape="12,200,3,3" size="86400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="306" name="conv7_2_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="307" name="conv7_2_mbox_conf/Dims6374/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4213008" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="308" name="conv7_2_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
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
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="309" name="conv7_2_mbox_conf_perm/Cast_117568_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="310" name="conv7_2_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="311" name="conv7_2_mbox_conf_flat/Cast_117678_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="312" name="conv7_2_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>216</dim>
				</port>
			</output>
		</layer>
		<layer id="313" name="353/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4213056" shape="8,224,3,3" size="64512"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>224</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="314" name="conv8_2_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>224</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="315" name="conv8_2_mbox_conf/Dims6398/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4277568" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="316" name="conv8_2_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
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
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="317" name="conv8_2_mbox_conf_perm/Cast_117716_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="318" name="conv8_2_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="319" name="conv8_2_mbox_conf_flat/Cast_117662_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="320" name="conv8_2_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
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
					<dim>48</dim>
				</port>
			</output>
		</layer>
		<layer id="321" name="328/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4277600" shape="8,128,3,3" size="36864"/>
			<output>
				<port id="1" precision="FP32">
					<dim>8</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="322" name="conv9_2_mbox_conf/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>8</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="323" name="conv9_2_mbox_conf/Dims6380/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4314464" shape="1,8,1,1" size="32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="324" name="conv9_2_mbox_conf" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
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
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="325" name="conv9_2_mbox_conf_perm/Cast_117712_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290256" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="326" name="conv9_2_mbox_conf_perm" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="327" name="conv9_2_mbox_conf_flat/Cast_117676_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="328" name="conv9_2_mbox_conf_flat" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
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
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="329" name="mbox_conf" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>792</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>216</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>48</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="8" precision="FP32">
					<dim>1</dim>
					<dim>23248</dim>
				</port>
			</output>
		</layer>
		<layer id="330" name="mbox_conf_reshape/Cast_117618_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314496" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="331" name="mbox_conf_reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>23248</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>11624</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="332" name="mbox_conf_softmax" type="SoftMax" version="opset1">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>11624</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>11624</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="333" name="mbox_conf_flatten/Cast_117652_const" type="Const" version="opset1">
			<data element_type="i64" offset="2290288" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="334" name="mbox_conf_flatten" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>11624</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>23248</dim>
				</port>
			</output>
		</layer>
		<layer id="335" name="conv4_3_0_norm_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="336" name="18781" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="337" name="conv4_3_0_norm_mbox_priorbox/ss_0_port/Cast_117706_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="338" name="conv4_3_0_norm_mbox_priorbox/ss_0_port/Cast_217708_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="339" name="conv4_3_0_norm_mbox_priorbox/ss_0_port/Cast_317710_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="340" name="conv4_3_0_norm_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="341" name="18745" type="Convert" version="opset1">
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
		<layer id="342" name="conv4_3_0_norm_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="343" name="18775" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="344" name="conv4_3_0_norm_mbox_priorbox/ss_1_port/Cast_117726_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="345" name="conv4_3_0_norm_mbox_priorbox/ss_1_port/Cast_217728_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="346" name="conv4_3_0_norm_mbox_priorbox/ss_1_port/Cast_317730_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="347" name="conv4_3_0_norm_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="348" name="18747" type="Convert" version="opset1">
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
		<layer id="349" name="conv4_3_0_norm_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="38.400001525878906" min_size="16.0" offset="0.5" step="16.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="350" name="conv4_3_0_norm_mbox_priorbox/unsqueeze/value14009_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="351" name="conv4_3_0_norm_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>16128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="352" name="conv4_3_norm_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="353" name="18797" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="354" name="conv4_3_norm_mbox_priorbox/ss_0_port/Cast_117670_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="355" name="conv4_3_norm_mbox_priorbox/ss_0_port/Cast_217672_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="356" name="conv4_3_norm_mbox_priorbox/ss_0_port/Cast_317674_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="357" name="conv4_3_norm_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="358" name="18757" type="Convert" version="opset1">
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
		<layer id="359" name="conv4_3_norm_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="360" name="18795" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="361" name="conv4_3_norm_mbox_priorbox/ss_1_port/Cast_117576_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="362" name="conv4_3_norm_mbox_priorbox/ss_1_port/Cast_217578_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="363" name="conv4_3_norm_mbox_priorbox/ss_1_port/Cast_317580_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="364" name="conv4_3_norm_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="365" name="18759" type="Convert" version="opset1">
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
		<layer id="366" name="conv4_3_norm_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="76.80000305175781" min_size="38.400001525878906" offset="0.5" step="16.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="367" name="conv4_3_norm_mbox_priorbox/unsqueeze/value13955_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="368" name="conv4_3_norm_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>16128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="369" name="fc7_0_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="370" name="18789" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="371" name="fc7_0_mbox_priorbox/ss_0_port/Cast_117646_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="372" name="fc7_0_mbox_priorbox/ss_0_port/Cast_217648_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="373" name="fc7_0_mbox_priorbox/ss_0_port/Cast_317650_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="374" name="fc7_0_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="375" name="18749" type="Convert" version="opset1">
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
		<layer id="376" name="fc7_0_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="377" name="18799" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="378" name="fc7_0_mbox_priorbox/ss_1_port/Cast_117654_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="379" name="fc7_0_mbox_priorbox/ss_1_port/Cast_217656_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="380" name="fc7_0_mbox_priorbox/ss_1_port/Cast_317658_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="381" name="fc7_0_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="382" name="18751" type="Convert" version="opset1">
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
		<layer id="383" name="fc7_0_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="124.80000305175781" min_size="76.80000305175781" offset="0.5" step="32.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="384" name="fc7_0_mbox_priorbox/unsqueeze/value14045_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="385" name="fc7_0_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>6048</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="386" name="fc7_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="387" name="18793" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="388" name="fc7_mbox_priorbox/ss_0_port/Cast_117610_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="389" name="fc7_mbox_priorbox/ss_0_port/Cast_217612_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="390" name="fc7_mbox_priorbox/ss_0_port/Cast_317614_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="391" name="fc7_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="392" name="18765" type="Convert" version="opset1">
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
		<layer id="393" name="fc7_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="394" name="18773" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="395" name="fc7_mbox_priorbox/ss_1_port/Cast_117588_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="396" name="fc7_mbox_priorbox/ss_1_port/Cast_217590_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="397" name="fc7_mbox_priorbox/ss_1_port/Cast_317592_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="398" name="fc7_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="399" name="18767" type="Convert" version="opset1">
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
		<layer id="400" name="fc7_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="172.8000030517578" min_size="124.80000305175781" offset="0.5" step="32.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="401" name="fc7_mbox_priorbox/unsqueeze/value14027_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="402" name="fc7_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>6048</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="403" name="conv6_2_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>152</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="404" name="18785" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="405" name="conv6_2_mbox_priorbox/ss_0_port/Cast_117718_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="406" name="conv6_2_mbox_priorbox/ss_0_port/Cast_217720_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="407" name="conv6_2_mbox_priorbox/ss_0_port/Cast_317722_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="408" name="conv6_2_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="409" name="18741" type="Convert" version="opset1">
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
		<layer id="410" name="conv6_2_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="411" name="18783" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="412" name="conv6_2_mbox_priorbox/ss_1_port/Cast_117690_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="413" name="conv6_2_mbox_priorbox/ss_1_port/Cast_217692_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="414" name="conv6_2_mbox_priorbox/ss_1_port/Cast_317694_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="415" name="conv6_2_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="416" name="18743" type="Convert" version="opset1">
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
		<layer id="417" name="conv6_2_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="230.39999389648438" min_size="172.8000030517578" offset="0.5" step="64.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="418" name="conv6_2_mbox_priorbox/unsqueeze/value13973_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="419" name="conv6_2_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>1584</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="420" name="conv7_2_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>200</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="421" name="18801" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="422" name="conv7_2_mbox_priorbox/ss_0_port/Cast_117664_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="423" name="conv7_2_mbox_priorbox/ss_0_port/Cast_217666_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="424" name="conv7_2_mbox_priorbox/ss_0_port/Cast_317668_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="425" name="conv7_2_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="426" name="18753" type="Convert" version="opset1">
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
		<layer id="427" name="conv7_2_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="428" name="18779" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="429" name="conv7_2_mbox_priorbox/ss_1_port/Cast_117640_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="430" name="conv7_2_mbox_priorbox/ss_1_port/Cast_217642_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="431" name="conv7_2_mbox_priorbox/ss_1_port/Cast_317644_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="432" name="conv7_2_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="433" name="18755" type="Convert" version="opset1">
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
		<layer id="434" name="conv7_2_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="288.0" min_size="230.39999389648438" offset="0.5" step="128.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="435" name="conv7_2_mbox_priorbox/unsqueeze/value14081_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="436" name="conv7_2_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>432</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="437" name="conv8_2_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="438" name="18777" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="439" name="conv8_2_mbox_priorbox/ss_0_port/Cast_117620_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="440" name="conv8_2_mbox_priorbox/ss_0_port/Cast_217622_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="441" name="conv8_2_mbox_priorbox/ss_0_port/Cast_317624_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="442" name="conv8_2_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="443" name="18769" type="Convert" version="opset1">
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
		<layer id="444" name="conv8_2_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="445" name="18803" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="446" name="conv8_2_mbox_priorbox/ss_1_port/Cast_117632_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="447" name="conv8_2_mbox_priorbox/ss_1_port/Cast_217634_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="448" name="conv8_2_mbox_priorbox/ss_1_port/Cast_317636_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="449" name="conv8_2_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="450" name="18771" type="Convert" version="opset1">
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
		<layer id="451" name="conv8_2_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="345.6000061035156" min_size="288.0" offset="0.5" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="452" name="conv8_2_mbox_priorbox/unsqueeze/value14063_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="453" name="conv8_2_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>96</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="454" name="conv9_2_mbox_priorbox/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="455" name="18787" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="456" name="conv9_2_mbox_priorbox/ss_0_port/Cast_117604_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="457" name="conv9_2_mbox_priorbox/ss_0_port/Cast_217606_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="458" name="conv9_2_mbox_priorbox/ss_0_port/Cast_317608_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="459" name="conv9_2_mbox_priorbox/ss_0_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="460" name="18761" type="Convert" version="opset1">
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
		<layer id="461" name="conv9_2_mbox_priorbox/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="462" name="18791" type="Convert" version="opset1">
			<data destination_type="i32"/>
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I32">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="463" name="conv9_2_mbox_priorbox/ss_1_port/Cast_117596_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314520" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="464" name="conv9_2_mbox_priorbox/ss_1_port/Cast_217598_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314528" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="465" name="conv9_2_mbox_priorbox/ss_1_port/Cast_317600_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="466" name="conv9_2_mbox_priorbox/ss_1_port" type="StridedSlice" version="opset1">
			<data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
			<input>
				<port id="0">
					<dim>4</dim>
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
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="467" name="18763" type="Convert" version="opset1">
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
		<layer id="468" name="conv9_2_mbox_priorbox/naked_not_unsqueezed" type="PriorBox" version="opset1">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="403.20001220703125" min_size="345.6000061035156" offset="0.5" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="469" name="conv9_2_mbox_priorbox/unsqueeze/value13991_const" type="Const" version="opset1">
			<data element_type="i64" offset="4314544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="470" name="conv9_2_mbox_priorbox" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="471" name="mbox_priorbox" type="Concat" version="opset1">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1584</dim>
				</port>
				<port id="5">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>2</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="8" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>46496</dim>
				</port>
			</output>
		</layer>
		<layer id="472" name="detection_out" type="DetectionOutput" version="opset1">
			<data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.009999999776482582" eta="1.0" input_height="1" input_width="1" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="1" num_classes="2" share_location="1" top_k="400" variance_encoded_in_target="0" visualize="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>46496</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>23248</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>46496</dim>
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
		<layer id="473" name="detection_out/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="19" from-port="1" to-layer="21" to-port="0"/>
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
		<edge from-layer="33" from-port="2" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="1" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="1"/>
		<edge from-layer="36" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="38" to-port="1"/>
		<edge from-layer="38" from-port="2" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="41" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="1"/>
		<edge from-layer="41" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="43" to-port="1"/>
		<edge from-layer="43" from-port="2" to-layer="44" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="46" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="46" to-port="1"/>
		<edge from-layer="46" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="48" to-port="1"/>
		<edge from-layer="48" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="51" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="51" to-port="1"/>
		<edge from-layer="51" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="52" from-port="1" to-layer="53" to-port="1"/>
		<edge from-layer="53" from-port="2" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="1" to-layer="56" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="56" to-port="1"/>
		<edge from-layer="56" from-port="2" to-layer="58" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="58" to-port="1"/>
		<edge from-layer="58" from-port="2" to-layer="59" to-port="0"/>
		<edge from-layer="59" from-port="1" to-layer="61" to-port="0"/>
		<edge from-layer="60" from-port="1" to-layer="61" to-port="1"/>
		<edge from-layer="61" from-port="2" to-layer="63" to-port="0"/>
		<edge from-layer="62" from-port="1" to-layer="63" to-port="1"/>
		<edge from-layer="63" from-port="2" to-layer="64" to-port="0"/>
		<edge from-layer="64" from-port="1" to-layer="66" to-port="0"/>
		<edge from-layer="65" from-port="1" to-layer="66" to-port="1"/>
		<edge from-layer="66" from-port="2" to-layer="68" to-port="0"/>
		<edge from-layer="67" from-port="1" to-layer="68" to-port="1"/>
		<edge from-layer="68" from-port="2" to-layer="69" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="71" to-port="0"/>
		<edge from-layer="70" from-port="1" to-layer="71" to-port="1"/>
		<edge from-layer="71" from-port="2" to-layer="73" to-port="0"/>
		<edge from-layer="72" from-port="1" to-layer="73" to-port="1"/>
		<edge from-layer="73" from-port="2" to-layer="74" to-port="0"/>
		<edge from-layer="74" from-port="1" to-layer="76" to-port="0"/>
		<edge from-layer="75" from-port="1" to-layer="76" to-port="1"/>
		<edge from-layer="76" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="77" from-port="1" to-layer="78" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="79" to-port="0"/>
		<edge from-layer="79" from-port="1" to-layer="81" to-port="0"/>
		<edge from-layer="80" from-port="1" to-layer="81" to-port="1"/>
		<edge from-layer="81" from-port="2" to-layer="83" to-port="0"/>
		<edge from-layer="82" from-port="1" to-layer="83" to-port="1"/>
		<edge from-layer="83" from-port="2" to-layer="84" to-port="0"/>
		<edge from-layer="84" from-port="1" to-layer="86" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="86" to-port="1"/>
		<edge from-layer="86" from-port="2" to-layer="88" to-port="0"/>
		<edge from-layer="87" from-port="1" to-layer="88" to-port="1"/>
		<edge from-layer="88" from-port="2" to-layer="89" to-port="0"/>
		<edge from-layer="89" from-port="1" to-layer="91" to-port="0"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="1"/>
		<edge from-layer="91" from-port="2" to-layer="93" to-port="0"/>
		<edge from-layer="92" from-port="1" to-layer="93" to-port="1"/>
		<edge from-layer="93" from-port="2" to-layer="94" to-port="0"/>
		<edge from-layer="94" from-port="1" to-layer="96" to-port="0"/>
		<edge from-layer="95" from-port="1" to-layer="96" to-port="1"/>
		<edge from-layer="96" from-port="2" to-layer="98" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="98" to-port="1"/>
		<edge from-layer="98" from-port="2" to-layer="99" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="101" to-port="0"/>
		<edge from-layer="100" from-port="1" to-layer="101" to-port="1"/>
		<edge from-layer="101" from-port="2" to-layer="103" to-port="0"/>
		<edge from-layer="102" from-port="1" to-layer="103" to-port="1"/>
		<edge from-layer="103" from-port="2" to-layer="104" to-port="0"/>
		<edge from-layer="104" from-port="1" to-layer="106" to-port="0"/>
		<edge from-layer="105" from-port="1" to-layer="106" to-port="1"/>
		<edge from-layer="106" from-port="2" to-layer="108" to-port="0"/>
		<edge from-layer="107" from-port="1" to-layer="108" to-port="1"/>
		<edge from-layer="108" from-port="2" to-layer="109" to-port="0"/>
		<edge from-layer="109" from-port="1" to-layer="111" to-port="0"/>
		<edge from-layer="110" from-port="1" to-layer="111" to-port="1"/>
		<edge from-layer="111" from-port="2" to-layer="113" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="113" to-port="1"/>
		<edge from-layer="113" from-port="2" to-layer="114" to-port="0"/>
		<edge from-layer="114" from-port="1" to-layer="116" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="116" to-port="1"/>
		<edge from-layer="116" from-port="2" to-layer="118" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="118" to-port="1"/>
		<edge from-layer="118" from-port="2" to-layer="119" to-port="0"/>
		<edge from-layer="119" from-port="1" to-layer="121" to-port="0"/>
		<edge from-layer="120" from-port="1" to-layer="121" to-port="1"/>
		<edge from-layer="121" from-port="2" to-layer="123" to-port="0"/>
		<edge from-layer="122" from-port="1" to-layer="123" to-port="1"/>
		<edge from-layer="123" from-port="2" to-layer="125" to-port="0"/>
		<edge from-layer="124" from-port="1" to-layer="125" to-port="1"/>
		<edge from-layer="125" from-port="2" to-layer="127" to-port="0"/>
		<edge from-layer="126" from-port="1" to-layer="127" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="129" to-port="0"/>
		<edge from-layer="128" from-port="1" to-layer="129" to-port="1"/>
		<edge from-layer="129" from-port="2" to-layer="131" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="131" to-port="1"/>
		<edge from-layer="131" from-port="2" to-layer="133" to-port="0"/>
		<edge from-layer="132" from-port="1" to-layer="133" to-port="1"/>
		<edge from-layer="133" from-port="2" to-layer="135" to-port="0"/>
		<edge from-layer="134" from-port="1" to-layer="135" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="137" to-port="0"/>
		<edge from-layer="136" from-port="1" to-layer="137" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="139" to-port="0"/>
		<edge from-layer="138" from-port="1" to-layer="139" to-port="1"/>
		<edge from-layer="139" from-port="2" to-layer="140" to-port="0"/>
		<edge from-layer="140" from-port="1" to-layer="142" to-port="0"/>
		<edge from-layer="141" from-port="1" to-layer="142" to-port="1"/>
		<edge from-layer="142" from-port="2" to-layer="144" to-port="0"/>
		<edge from-layer="143" from-port="1" to-layer="144" to-port="1"/>
		<edge from-layer="144" from-port="2" to-layer="145" to-port="0"/>
		<edge from-layer="145" from-port="1" to-layer="147" to-port="0"/>
		<edge from-layer="146" from-port="1" to-layer="147" to-port="1"/>
		<edge from-layer="147" from-port="2" to-layer="149" to-port="0"/>
		<edge from-layer="148" from-port="1" to-layer="149" to-port="1"/>
		<edge from-layer="149" from-port="2" to-layer="150" to-port="0"/>
		<edge from-layer="150" from-port="1" to-layer="152" to-port="0"/>
		<edge from-layer="151" from-port="1" to-layer="152" to-port="1"/>
		<edge from-layer="152" from-port="2" to-layer="154" to-port="0"/>
		<edge from-layer="153" from-port="1" to-layer="154" to-port="1"/>
		<edge from-layer="154" from-port="2" to-layer="155" to-port="0"/>
		<edge from-layer="155" from-port="1" to-layer="157" to-port="0"/>
		<edge from-layer="156" from-port="1" to-layer="157" to-port="1"/>
		<edge from-layer="157" from-port="2" to-layer="159" to-port="0"/>
		<edge from-layer="158" from-port="1" to-layer="159" to-port="1"/>
		<edge from-layer="159" from-port="2" to-layer="161" to-port="0"/>
		<edge from-layer="160" from-port="1" to-layer="161" to-port="1"/>
		<edge from-layer="161" from-port="2" to-layer="163" to-port="0"/>
		<edge from-layer="162" from-port="1" to-layer="163" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="165" to-port="0"/>
		<edge from-layer="164" from-port="1" to-layer="165" to-port="1"/>
		<edge from-layer="165" from-port="2" to-layer="167" to-port="0"/>
		<edge from-layer="166" from-port="1" to-layer="167" to-port="1"/>
		<edge from-layer="167" from-port="2" to-layer="169" to-port="0"/>
		<edge from-layer="168" from-port="1" to-layer="169" to-port="1"/>
		<edge from-layer="169" from-port="2" to-layer="171" to-port="0"/>
		<edge from-layer="170" from-port="1" to-layer="171" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="173" to-port="0"/>
		<edge from-layer="172" from-port="1" to-layer="173" to-port="1"/>
		<edge from-layer="173" from-port="2" to-layer="175" to-port="0"/>
		<edge from-layer="174" from-port="1" to-layer="175" to-port="1"/>
		<edge from-layer="175" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="176" from-port="1" to-layer="178" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="178" to-port="1"/>
		<edge from-layer="178" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="179" from-port="1" to-layer="180" to-port="1"/>
		<edge from-layer="180" from-port="2" to-layer="181" to-port="0"/>
		<edge from-layer="181" from-port="1" to-layer="183" to-port="0"/>
		<edge from-layer="182" from-port="1" to-layer="183" to-port="1"/>
		<edge from-layer="183" from-port="2" to-layer="185" to-port="0"/>
		<edge from-layer="184" from-port="1" to-layer="185" to-port="1"/>
		<edge from-layer="185" from-port="2" to-layer="186" to-port="0"/>
		<edge from-layer="186" from-port="1" to-layer="188" to-port="0"/>
		<edge from-layer="187" from-port="1" to-layer="188" to-port="1"/>
		<edge from-layer="188" from-port="2" to-layer="190" to-port="0"/>
		<edge from-layer="189" from-port="1" to-layer="190" to-port="1"/>
		<edge from-layer="190" from-port="2" to-layer="192" to-port="0"/>
		<edge from-layer="191" from-port="1" to-layer="192" to-port="1"/>
		<edge from-layer="192" from-port="2" to-layer="194" to-port="0"/>
		<edge from-layer="193" from-port="1" to-layer="194" to-port="1"/>
		<edge from-layer="186" from-port="1" to-layer="196" to-port="0"/>
		<edge from-layer="195" from-port="1" to-layer="196" to-port="1"/>
		<edge from-layer="196" from-port="2" to-layer="198" to-port="0"/>
		<edge from-layer="197" from-port="1" to-layer="198" to-port="1"/>
		<edge from-layer="198" from-port="2" to-layer="199" to-port="0"/>
		<edge from-layer="199" from-port="1" to-layer="201" to-port="0"/>
		<edge from-layer="200" from-port="1" to-layer="201" to-port="1"/>
		<edge from-layer="201" from-port="2" to-layer="203" to-port="0"/>
		<edge from-layer="202" from-port="1" to-layer="203" to-port="1"/>
		<edge from-layer="203" from-port="2" to-layer="204" to-port="0"/>
		<edge from-layer="204" from-port="1" to-layer="206" to-port="0"/>
		<edge from-layer="205" from-port="1" to-layer="206" to-port="1"/>
		<edge from-layer="206" from-port="2" to-layer="208" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="208" to-port="1"/>
		<edge from-layer="208" from-port="2" to-layer="209" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="211" to-port="0"/>
		<edge from-layer="210" from-port="1" to-layer="211" to-port="1"/>
		<edge from-layer="211" from-port="2" to-layer="213" to-port="0"/>
		<edge from-layer="212" from-port="1" to-layer="213" to-port="1"/>
		<edge from-layer="213" from-port="2" to-layer="215" to-port="0"/>
		<edge from-layer="214" from-port="1" to-layer="215" to-port="1"/>
		<edge from-layer="215" from-port="2" to-layer="217" to-port="0"/>
		<edge from-layer="216" from-port="1" to-layer="217" to-port="1"/>
		<edge from-layer="209" from-port="1" to-layer="219" to-port="0"/>
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
		<edge from-layer="232" from-port="1" to-layer="234" to-port="0"/>
		<edge from-layer="233" from-port="1" to-layer="234" to-port="1"/>
		<edge from-layer="234" from-port="2" to-layer="236" to-port="0"/>
		<edge from-layer="235" from-port="1" to-layer="236" to-port="1"/>
		<edge from-layer="236" from-port="2" to-layer="238" to-port="0"/>
		<edge from-layer="237" from-port="1" to-layer="238" to-port="1"/>
		<edge from-layer="238" from-port="2" to-layer="240" to-port="0"/>
		<edge from-layer="239" from-port="1" to-layer="240" to-port="1"/>
		<edge from-layer="232" from-port="1" to-layer="242" to-port="0"/>
		<edge from-layer="241" from-port="1" to-layer="242" to-port="1"/>
		<edge from-layer="242" from-port="2" to-layer="244" to-port="0"/>
		<edge from-layer="243" from-port="1" to-layer="244" to-port="1"/>
		<edge from-layer="244" from-port="2" to-layer="245" to-port="0"/>
		<edge from-layer="245" from-port="1" to-layer="247" to-port="0"/>
		<edge from-layer="246" from-port="1" to-layer="247" to-port="1"/>
		<edge from-layer="247" from-port="2" to-layer="249" to-port="0"/>
		<edge from-layer="248" from-port="1" to-layer="249" to-port="1"/>
		<edge from-layer="249" from-port="2" to-layer="250" to-port="0"/>
		<edge from-layer="250" from-port="1" to-layer="252" to-port="0"/>
		<edge from-layer="251" from-port="1" to-layer="252" to-port="1"/>
		<edge from-layer="252" from-port="2" to-layer="254" to-port="0"/>
		<edge from-layer="253" from-port="1" to-layer="254" to-port="1"/>
		<edge from-layer="254" from-port="2" to-layer="255" to-port="0"/>
		<edge from-layer="255" from-port="1" to-layer="257" to-port="0"/>
		<edge from-layer="256" from-port="1" to-layer="257" to-port="1"/>
		<edge from-layer="257" from-port="2" to-layer="259" to-port="0"/>
		<edge from-layer="258" from-port="1" to-layer="259" to-port="1"/>
		<edge from-layer="259" from-port="2" to-layer="261" to-port="0"/>
		<edge from-layer="260" from-port="1" to-layer="261" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="263" to-port="0"/>
		<edge from-layer="262" from-port="1" to-layer="263" to-port="1"/>
		<edge from-layer="127" from-port="2" to-layer="264" to-port="0"/>
		<edge from-layer="135" from-port="2" to-layer="264" to-port="1"/>
		<edge from-layer="163" from-port="2" to-layer="264" to-port="2"/>
		<edge from-layer="171" from-port="2" to-layer="264" to-port="3"/>
		<edge from-layer="194" from-port="2" to-layer="264" to-port="4"/>
		<edge from-layer="217" from-port="2" to-layer="264" to-port="5"/>
		<edge from-layer="240" from-port="2" to-layer="264" to-port="6"/>
		<edge from-layer="263" from-port="2" to-layer="264" to-port="7"/>
		<edge from-layer="119" from-port="1" to-layer="266" to-port="0"/>
		<edge from-layer="265" from-port="1" to-layer="266" to-port="1"/>
		<edge from-layer="266" from-port="2" to-layer="268" to-port="0"/>
		<edge from-layer="267" from-port="1" to-layer="268" to-port="1"/>
		<edge from-layer="268" from-port="2" to-layer="270" to-port="0"/>
		<edge from-layer="269" from-port="1" to-layer="270" to-port="1"/>
		<edge from-layer="270" from-port="2" to-layer="272" to-port="0"/>
		<edge from-layer="271" from-port="1" to-layer="272" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="274" to-port="0"/>
		<edge from-layer="273" from-port="1" to-layer="274" to-port="1"/>
		<edge from-layer="274" from-port="2" to-layer="276" to-port="0"/>
		<edge from-layer="275" from-port="1" to-layer="276" to-port="1"/>
		<edge from-layer="276" from-port="2" to-layer="278" to-port="0"/>
		<edge from-layer="277" from-port="1" to-layer="278" to-port="1"/>
		<edge from-layer="278" from-port="2" to-layer="280" to-port="0"/>
		<edge from-layer="279" from-port="1" to-layer="280" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="282" to-port="0"/>
		<edge from-layer="281" from-port="1" to-layer="282" to-port="1"/>
		<edge from-layer="282" from-port="2" to-layer="284" to-port="0"/>
		<edge from-layer="283" from-port="1" to-layer="284" to-port="1"/>
		<edge from-layer="284" from-port="2" to-layer="286" to-port="0"/>
		<edge from-layer="285" from-port="1" to-layer="286" to-port="1"/>
		<edge from-layer="286" from-port="2" to-layer="288" to-port="0"/>
		<edge from-layer="287" from-port="1" to-layer="288" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="290" to-port="0"/>
		<edge from-layer="289" from-port="1" to-layer="290" to-port="1"/>
		<edge from-layer="290" from-port="2" to-layer="292" to-port="0"/>
		<edge from-layer="291" from-port="1" to-layer="292" to-port="1"/>
		<edge from-layer="292" from-port="2" to-layer="294" to-port="0"/>
		<edge from-layer="293" from-port="1" to-layer="294" to-port="1"/>
		<edge from-layer="294" from-port="2" to-layer="296" to-port="0"/>
		<edge from-layer="295" from-port="1" to-layer="296" to-port="1"/>
		<edge from-layer="186" from-port="1" to-layer="298" to-port="0"/>
		<edge from-layer="297" from-port="1" to-layer="298" to-port="1"/>
		<edge from-layer="298" from-port="2" to-layer="300" to-port="0"/>
		<edge from-layer="299" from-port="1" to-layer="300" to-port="1"/>
		<edge from-layer="300" from-port="2" to-layer="302" to-port="0"/>
		<edge from-layer="301" from-port="1" to-layer="302" to-port="1"/>
		<edge from-layer="302" from-port="2" to-layer="304" to-port="0"/>
		<edge from-layer="303" from-port="1" to-layer="304" to-port="1"/>
		<edge from-layer="209" from-port="1" to-layer="306" to-port="0"/>
		<edge from-layer="305" from-port="1" to-layer="306" to-port="1"/>
		<edge from-layer="306" from-port="2" to-layer="308" to-port="0"/>
		<edge from-layer="307" from-port="1" to-layer="308" to-port="1"/>
		<edge from-layer="308" from-port="2" to-layer="310" to-port="0"/>
		<edge from-layer="309" from-port="1" to-layer="310" to-port="1"/>
		<edge from-layer="310" from-port="2" to-layer="312" to-port="0"/>
		<edge from-layer="311" from-port="1" to-layer="312" to-port="1"/>
		<edge from-layer="232" from-port="1" to-layer="314" to-port="0"/>
		<edge from-layer="313" from-port="1" to-layer="314" to-port="1"/>
		<edge from-layer="314" from-port="2" to-layer="316" to-port="0"/>
		<edge from-layer="315" from-port="1" to-layer="316" to-port="1"/>
		<edge from-layer="316" from-port="2" to-layer="318" to-port="0"/>
		<edge from-layer="317" from-port="1" to-layer="318" to-port="1"/>
		<edge from-layer="318" from-port="2" to-layer="320" to-port="0"/>
		<edge from-layer="319" from-port="1" to-layer="320" to-port="1"/>
		<edge from-layer="255" from-port="1" to-layer="322" to-port="0"/>
		<edge from-layer="321" from-port="1" to-layer="322" to-port="1"/>
		<edge from-layer="322" from-port="2" to-layer="324" to-port="0"/>
		<edge from-layer="323" from-port="1" to-layer="324" to-port="1"/>
		<edge from-layer="324" from-port="2" to-layer="326" to-port="0"/>
		<edge from-layer="325" from-port="1" to-layer="326" to-port="1"/>
		<edge from-layer="326" from-port="2" to-layer="328" to-port="0"/>
		<edge from-layer="327" from-port="1" to-layer="328" to-port="1"/>
		<edge from-layer="272" from-port="2" to-layer="329" to-port="0"/>
		<edge from-layer="280" from-port="2" to-layer="329" to-port="1"/>
		<edge from-layer="288" from-port="2" to-layer="329" to-port="2"/>
		<edge from-layer="296" from-port="2" to-layer="329" to-port="3"/>
		<edge from-layer="304" from-port="2" to-layer="329" to-port="4"/>
		<edge from-layer="312" from-port="2" to-layer="329" to-port="5"/>
		<edge from-layer="320" from-port="2" to-layer="329" to-port="6"/>
		<edge from-layer="328" from-port="2" to-layer="329" to-port="7"/>
		<edge from-layer="329" from-port="8" to-layer="331" to-port="0"/>
		<edge from-layer="330" from-port="1" to-layer="331" to-port="1"/>
		<edge from-layer="331" from-port="2" to-layer="332" to-port="0"/>
		<edge from-layer="332" from-port="1" to-layer="334" to-port="0"/>
		<edge from-layer="333" from-port="1" to-layer="334" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="335" to-port="0"/>
		<edge from-layer="335" from-port="1" to-layer="336" to-port="0"/>
		<edge from-layer="336" from-port="1" to-layer="340" to-port="0"/>
		<edge from-layer="337" from-port="1" to-layer="340" to-port="1"/>
		<edge from-layer="338" from-port="1" to-layer="340" to-port="2"/>
		<edge from-layer="339" from-port="1" to-layer="340" to-port="3"/>
		<edge from-layer="340" from-port="4" to-layer="341" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="342" to-port="0"/>
		<edge from-layer="342" from-port="1" to-layer="343" to-port="0"/>
		<edge from-layer="343" from-port="1" to-layer="347" to-port="0"/>
		<edge from-layer="344" from-port="1" to-layer="347" to-port="1"/>
		<edge from-layer="345" from-port="1" to-layer="347" to-port="2"/>
		<edge from-layer="346" from-port="1" to-layer="347" to-port="3"/>
		<edge from-layer="347" from-port="4" to-layer="348" to-port="0"/>
		<edge from-layer="341" from-port="1" to-layer="349" to-port="0"/>
		<edge from-layer="348" from-port="1" to-layer="349" to-port="1"/>
		<edge from-layer="349" from-port="2" to-layer="351" to-port="0"/>
		<edge from-layer="350" from-port="1" to-layer="351" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="352" to-port="0"/>
		<edge from-layer="352" from-port="1" to-layer="353" to-port="0"/>
		<edge from-layer="353" from-port="1" to-layer="357" to-port="0"/>
		<edge from-layer="354" from-port="1" to-layer="357" to-port="1"/>
		<edge from-layer="355" from-port="1" to-layer="357" to-port="2"/>
		<edge from-layer="356" from-port="1" to-layer="357" to-port="3"/>
		<edge from-layer="357" from-port="4" to-layer="358" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="359" to-port="0"/>
		<edge from-layer="359" from-port="1" to-layer="360" to-port="0"/>
		<edge from-layer="360" from-port="1" to-layer="364" to-port="0"/>
		<edge from-layer="361" from-port="1" to-layer="364" to-port="1"/>
		<edge from-layer="362" from-port="1" to-layer="364" to-port="2"/>
		<edge from-layer="363" from-port="1" to-layer="364" to-port="3"/>
		<edge from-layer="364" from-port="4" to-layer="365" to-port="0"/>
		<edge from-layer="358" from-port="1" to-layer="366" to-port="0"/>
		<edge from-layer="365" from-port="1" to-layer="366" to-port="1"/>
		<edge from-layer="366" from-port="2" to-layer="368" to-port="0"/>
		<edge from-layer="367" from-port="1" to-layer="368" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="369" to-port="0"/>
		<edge from-layer="369" from-port="1" to-layer="370" to-port="0"/>
		<edge from-layer="370" from-port="1" to-layer="374" to-port="0"/>
		<edge from-layer="371" from-port="1" to-layer="374" to-port="1"/>
		<edge from-layer="372" from-port="1" to-layer="374" to-port="2"/>
		<edge from-layer="373" from-port="1" to-layer="374" to-port="3"/>
		<edge from-layer="374" from-port="4" to-layer="375" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="376" to-port="0"/>
		<edge from-layer="376" from-port="1" to-layer="377" to-port="0"/>
		<edge from-layer="377" from-port="1" to-layer="381" to-port="0"/>
		<edge from-layer="378" from-port="1" to-layer="381" to-port="1"/>
		<edge from-layer="379" from-port="1" to-layer="381" to-port="2"/>
		<edge from-layer="380" from-port="1" to-layer="381" to-port="3"/>
		<edge from-layer="381" from-port="4" to-layer="382" to-port="0"/>
		<edge from-layer="375" from-port="1" to-layer="383" to-port="0"/>
		<edge from-layer="382" from-port="1" to-layer="383" to-port="1"/>
		<edge from-layer="383" from-port="2" to-layer="385" to-port="0"/>
		<edge from-layer="384" from-port="1" to-layer="385" to-port="1"/>
		<edge from-layer="155" from-port="1" to-layer="386" to-port="0"/>
		<edge from-layer="386" from-port="1" to-layer="387" to-port="0"/>
		<edge from-layer="387" from-port="1" to-layer="391" to-port="0"/>
		<edge from-layer="388" from-port="1" to-layer="391" to-port="1"/>
		<edge from-layer="389" from-port="1" to-layer="391" to-port="2"/>
		<edge from-layer="390" from-port="1" to-layer="391" to-port="3"/>
		<edge from-layer="391" from-port="4" to-layer="392" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="393" to-port="0"/>
		<edge from-layer="393" from-port="1" to-layer="394" to-port="0"/>
		<edge from-layer="394" from-port="1" to-layer="398" to-port="0"/>
		<edge from-layer="395" from-port="1" to-layer="398" to-port="1"/>
		<edge from-layer="396" from-port="1" to-layer="398" to-port="2"/>
		<edge from-layer="397" from-port="1" to-layer="398" to-port="3"/>
		<edge from-layer="398" from-port="4" to-layer="399" to-port="0"/>
		<edge from-layer="392" from-port="1" to-layer="400" to-port="0"/>
		<edge from-layer="399" from-port="1" to-layer="400" to-port="1"/>
		<edge from-layer="400" from-port="2" to-layer="402" to-port="0"/>
		<edge from-layer="401" from-port="1" to-layer="402" to-port="1"/>
		<edge from-layer="186" from-port="1" to-layer="403" to-port="0"/>
		<edge from-layer="403" from-port="1" to-layer="404" to-port="0"/>
		<edge from-layer="404" from-port="1" to-layer="408" to-port="0"/>
		<edge from-layer="405" from-port="1" to-layer="408" to-port="1"/>
		<edge from-layer="406" from-port="1" to-layer="408" to-port="2"/>
		<edge from-layer="407" from-port="1" to-layer="408" to-port="3"/>
		<edge from-layer="408" from-port="4" to-layer="409" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="410" to-port="0"/>
		<edge from-layer="410" from-port="1" to-layer="411" to-port="0"/>
		<edge from-layer="411" from-port="1" to-layer="415" to-port="0"/>
		<edge from-layer="412" from-port="1" to-layer="415" to-port="1"/>
		<edge from-layer="413" from-port="1" to-layer="415" to-port="2"/>
		<edge from-layer="414" from-port="1" to-layer="415" to-port="3"/>
		<edge from-layer="415" from-port="4" to-layer="416" to-port="0"/>
		<edge from-layer="409" from-port="1" to-layer="417" to-port="0"/>
		<edge from-layer="416" from-port="1" to-layer="417" to-port="1"/>
		<edge from-layer="417" from-port="2" to-layer="419" to-port="0"/>
		<edge from-layer="418" from-port="1" to-layer="419" to-port="1"/>
		<edge from-layer="209" from-port="1" to-layer="420" to-port="0"/>
		<edge from-layer="420" from-port="1" to-layer="421" to-port="0"/>
		<edge from-layer="421" from-port="1" to-layer="425" to-port="0"/>
		<edge from-layer="422" from-port="1" to-layer="425" to-port="1"/>
		<edge from-layer="423" from-port="1" to-layer="425" to-port="2"/>
		<edge from-layer="424" from-port="1" to-layer="425" to-port="3"/>
		<edge from-layer="425" from-port="4" to-layer="426" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="427" to-port="0"/>
		<edge from-layer="427" from-port="1" to-layer="428" to-port="0"/>
		<edge from-layer="428" from-port="1" to-layer="432" to-port="0"/>
		<edge from-layer="429" from-port="1" to-layer="432" to-port="1"/>
		<edge from-layer="430" from-port="1" to-layer="432" to-port="2"/>
		<edge from-layer="431" from-port="1" to-layer="432" to-port="3"/>
		<edge from-layer="432" from-port="4" to-layer="433" to-port="0"/>
		<edge from-layer="426" from-port="1" to-layer="434" to-port="0"/>
		<edge from-layer="433" from-port="1" to-layer="434" to-port="1"/>
		<edge from-layer="434" from-port="2" to-layer="436" to-port="0"/>
		<edge from-layer="435" from-port="1" to-layer="436" to-port="1"/>
		<edge from-layer="232" from-port="1" to-layer="437" to-port="0"/>
		<edge from-layer="437" from-port="1" to-layer="438" to-port="0"/>
		<edge from-layer="438" from-port="1" to-layer="442" to-port="0"/>
		<edge from-layer="439" from-port="1" to-layer="442" to-port="1"/>
		<edge from-layer="440" from-port="1" to-layer="442" to-port="2"/>
		<edge from-layer="441" from-port="1" to-layer="442" to-port="3"/>
		<edge from-layer="442" from-port="4" to-layer="443" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="444" to-port="0"/>
		<edge from-layer="444" from-port="1" to-layer="445" to-port="0"/>
		<edge from-layer="445" from-port="1" to-layer="449" to-port="0"/>
		<edge from-layer="446" from-port="1" to-layer="449" to-port="1"/>
		<edge from-layer="447" from-port="1" to-layer="449" to-port="2"/>
		<edge from-layer="448" from-port="1" to-layer="449" to-port="3"/>
		<edge from-layer="449" from-port="4" to-layer="450" to-port="0"/>
		<edge from-layer="443" from-port="1" to-layer="451" to-port="0"/>
		<edge from-layer="450" from-port="1" to-layer="451" to-port="1"/>
		<edge from-layer="451" from-port="2" to-layer="453" to-port="0"/>
		<edge from-layer="452" from-port="1" to-layer="453" to-port="1"/>
		<edge from-layer="255" from-port="1" to-layer="454" to-port="0"/>
		<edge from-layer="454" from-port="1" to-layer="455" to-port="0"/>
		<edge from-layer="455" from-port="1" to-layer="459" to-port="0"/>
		<edge from-layer="456" from-port="1" to-layer="459" to-port="1"/>
		<edge from-layer="457" from-port="1" to-layer="459" to-port="2"/>
		<edge from-layer="458" from-port="1" to-layer="459" to-port="3"/>
		<edge from-layer="459" from-port="4" to-layer="460" to-port="0"/>
		<edge from-layer="4" from-port="2" to-layer="461" to-port="0"/>
		<edge from-layer="461" from-port="1" to-layer="462" to-port="0"/>
		<edge from-layer="462" from-port="1" to-layer="466" to-port="0"/>
		<edge from-layer="463" from-port="1" to-layer="466" to-port="1"/>
		<edge from-layer="464" from-port="1" to-layer="466" to-port="2"/>
		<edge from-layer="465" from-port="1" to-layer="466" to-port="3"/>
		<edge from-layer="466" from-port="4" to-layer="467" to-port="0"/>
		<edge from-layer="460" from-port="1" to-layer="468" to-port="0"/>
		<edge from-layer="467" from-port="1" to-layer="468" to-port="1"/>
		<edge from-layer="468" from-port="2" to-layer="470" to-port="0"/>
		<edge from-layer="469" from-port="1" to-layer="470" to-port="1"/>
		<edge from-layer="351" from-port="2" to-layer="471" to-port="0"/>
		<edge from-layer="368" from-port="2" to-layer="471" to-port="1"/>
		<edge from-layer="385" from-port="2" to-layer="471" to-port="2"/>
		<edge from-layer="402" from-port="2" to-layer="471" to-port="3"/>
		<edge from-layer="419" from-port="2" to-layer="471" to-port="4"/>
		<edge from-layer="436" from-port="2" to-layer="471" to-port="5"/>
		<edge from-layer="453" from-port="2" to-layer="471" to-port="6"/>
		<edge from-layer="470" from-port="2" to-layer="471" to-port="7"/>
		<edge from-layer="264" from-port="8" to-layer="472" to-port="0"/>
		<edge from-layer="334" from-port="2" to-layer="472" to-port="1"/>
		<edge from-layer="471" from-port="8" to-layer="472" to-port="2"/>
		<edge from-layer="472" from-port="3" to-layer="473" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="0.0.0-2658-g49d9842a1"/>
		<cli_parameters>
			<blobs_as_inputs value="True"/>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP32"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="caffe"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V2 value="False"/>
			<generate_deprecated_IR_V7 value="False"/>
			<generate_experimental_IR_V10 value="True"/>
			<input value="data"/>
			<input_model value="DIR/mobilenet384x672-reduced-vd.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/mobilenet384x672-reduced-vd.prototxt"/>
			<input_shape value="[1,3,384,672]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'mean': array([104., 117., 123.]), 'scale': array([58.82352941])}}"/>
			<mean_values value="data[104.0,117.0,123.0]"/>
			<model_name value="vehicle-detection-adas-0002"/>
			<move_to_preprocess value="False"/>
			<output value="['detection_out']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'data': array([  1,   3, 384, 672])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="data[58.8235294117647]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
