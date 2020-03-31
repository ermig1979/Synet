<?xml version="1.0" ?>
<net name="face-detection-retail-0005" version="10">
	<layers>
		<layer id="0" name="input.1" type="Parameter" version="opset1">
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
		<layer id="1" name="data_mul_/copy_const" type="Const" version="opset1">
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
		<layer id="2" name="Mul_" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
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
					<dim>300</dim>
					<dim>300</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="input.1/reverse_input_channels13842/Concat13861_const" type="Const" version="opset1">
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
		<layer id="4" name="Mul1_8473/Fused_Mul_" type="Multiply" version="opset1">
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
		<layer id="5" name="input.1/reverse_input_channels13844/Concat13853_const" type="Const" version="opset1">
			<data element_type="f32" offset="16" shape="1,3,1,1" size="12"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="Add1_8474/Fused_Add_" type="Add" version="opset1">
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
		<layer id="7" name="input.1/reverse_input_channels/Concat13869_const" type="Const" version="opset1">
			<data element_type="f32" offset="28" shape="24,3,3,3" size="2592"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="332" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="data_add_91069111/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2620" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="Add1_8042/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
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
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="334" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="1430514308_const" type="Const" version="opset1">
			<data element_type="f32" offset="2716" shape="24,1,1,3,3" size="864"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="335" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="data_add_91149119/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3580" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="Add1_8510/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
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
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="337" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="Mul1_8293/Fused_Mul_99939995_const" type="Const" version="opset1">
			<data element_type="f32" offset="3676" shape="12,24,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>12</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="338" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>12</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="data_add_91229127/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4828" shape="1,12,1,1" size="48"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="Add1_8294/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>150</dim>
					<dim>150</dim>
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
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="Mul1_8317/Fused_Mul_99979999_const" type="Const" version="opset1">
			<data element_type="f32" offset="4876" shape="72,12,1,1" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>72</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="340" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>72</dim>
					<dim>12</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="data_add_91309135/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8332" shape="1,72,1,1" size="288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="Add1_8318/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="342" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="1431714320_const" type="Const" version="opset1">
			<data element_type="f32" offset="8620" shape="72,1,1,3,3" size="2592"/>
			<output>
				<port id="1" precision="FP32">
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="343" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>150</dim>
					<dim>150</dim>
				</port>
				<port id="1">
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="data_add_91389143/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11212" shape="1,72,1,1" size="288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="Add1_8366/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="345" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="Mul1_8221/Fused_Mul_1000510007_const" type="Const" version="opset1">
			<data element_type="f32" offset="11500" shape="18,72,1,1" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>18</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="346" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>18</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="data_add_91469151/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="16684" shape="1,18,1,1" size="72"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="Add1_8222/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="Mul1_8089/Fused_Mul_1000910011_const" type="Const" version="opset1">
			<data element_type="f32" offset="16756" shape="108,18,1,1" size="7776"/>
			<output>
				<port id="1" precision="FP32">
					<dim>108</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="348" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>108</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="data_add_91549159/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="24532" shape="1,108,1,1" size="432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="Add1_8090/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="350" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="1430914312_const" type="Const" version="opset1">
			<data element_type="f32" offset="24964" shape="108,1,1,3,3" size="3888"/>
			<output>
				<port id="1" precision="FP32">
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="351" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="data_add_91629167/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="28852" shape="1,108,1,1" size="432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="Add1_8174/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="353" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="Mul1_7957/Fused_Mul_1001710019_const" type="Const" version="opset1">
			<data element_type="f32" offset="29284" shape="18,108,1,1" size="7776"/>
			<output>
				<port id="1" precision="FP32">
					<dim>18</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="354" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>18</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="data_add_91709175/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="37060" shape="1,18,1,1" size="72"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="Add1_7958/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="356" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="Mul1_8245/Fused_Mul_1002110023_const" type="Const" version="opset1">
			<data element_type="f32" offset="37132" shape="108,18,1,1" size="7776"/>
			<output>
				<port id="1" precision="FP32">
					<dim>108</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="357" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>108</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="data_add_91789183/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="44908" shape="1,108,1,1" size="432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="Add1_8246/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="359" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="1433714340_const" type="Const" version="opset1">
			<data element_type="f32" offset="45340" shape="108,1,1,3,3" size="3888"/>
			<output>
				<port id="1" precision="FP32">
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="360" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>75</dim>
					<dim>75</dim>
				</port>
				<port id="1">
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="data_add_91869191/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="49228" shape="1,108,1,1" size="432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="Add1_8234/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="362" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="Mul1_/Fused_Mul_1002910031_const" type="Const" version="opset1">
			<data element_type="f32" offset="49660" shape="24,108,1,1" size="10368"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="363" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>108</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>24</dim>
					<dim>108</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="data_add_91949199/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="60028" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="Add1_/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="Mul1_8137/Fused_Mul_1003310035_const" type="Const" version="opset1">
			<data element_type="f32" offset="60124" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="365" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="data_add_92029207/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="73948" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="Add1_8138/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="367" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="1428114284_const" type="Const" version="opset1">
			<data element_type="f32" offset="74524" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="70" name="368" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="data_add_92109215/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="79708" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="Add1_8438/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="370" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="Mul1_8125/Fused_Mul_1004110043_const" type="Const" version="opset1">
			<data element_type="f32" offset="80284" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="371" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="data_add_92189223/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="94108" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="Add1_8126/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="373" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="Mul1_8401/Fused_Mul_1004510047_const" type="Const" version="opset1">
			<data element_type="f32" offset="94204" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="374" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="data_add_92269231/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="108028" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="Add1_8402/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="376" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="1428514288_const" type="Const" version="opset1">
			<data element_type="f32" offset="108604" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="85" name="377" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="data_add_92349239/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="113788" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="Add1_8258/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="379" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="Mul1_7969/Fused_Mul_1005310055_const" type="Const" version="opset1">
			<data element_type="f32" offset="114364" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="380" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="data_add_92429247/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="128188" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="Add1_7970/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="382" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="Mul1_8305/Fused_Mul_1005710059_const" type="Const" version="opset1">
			<data element_type="f32" offset="128284" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="383" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="data_add_92509255/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="142108" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="Add1_8306/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="385" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="1429314296_const" type="Const" version="opset1">
			<data element_type="f32" offset="142684" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="100" name="386" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>38</dim>
					<dim>38</dim>
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
		<layer id="101" name="data_add_92589263/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="147868" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="Add1_8018/Fused_Add_" type="Add" version="opset1">
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
		<layer id="103" name="388" type="ReLU" version="opset1">
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
		<layer id="104" name="Mul1_8353/Fused_Mul_1006510067_const" type="Const" version="opset1">
			<data element_type="f32" offset="148444" shape="48,144,1,1" size="27648"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="389" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="data_add_92669271/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="176092" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="Add1_8354/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="Mul1_7993/Fused_Mul_1006910071_const" type="Const" version="opset1">
			<data element_type="f32" offset="176284" shape="288,48,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="391" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="data_add_92749279/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="231580" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="Add1_7994/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="393" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="1428914292_const" type="Const" version="opset1">
			<data element_type="f32" offset="232732" shape="288,1,1,3,3" size="10368"/>
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
		<layer id="114" name="394" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="data_add_92829287/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="243100" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="Add1_8186/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="396" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="Mul1_8005/Fused_Mul_1007710079_const" type="Const" version="opset1">
			<data element_type="f32" offset="244252" shape="48,288,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="397" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="data_add_92909295/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="299548" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="Add1_8006/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="399" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="Mul1_8269/Fused_Mul_1008110083_const" type="Const" version="opset1">
			<data element_type="f32" offset="299740" shape="288,48,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="400" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="data_add_92989303/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="355036" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="Add1_8270/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="402" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="1434514348_const" type="Const" version="opset1">
			<data element_type="f32" offset="356188" shape="288,1,1,3,3" size="10368"/>
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
		<layer id="129" name="403" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="data_add_93069311/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="366556" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="Add1_8150/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="405" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="Mul1_8197/Fused_Mul_1008910091_const" type="Const" version="opset1">
			<data element_type="f32" offset="367708" shape="48,288,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="406" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="data_add_93149319/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="423004" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="Add1_8198/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="408" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="Mul1_8209/Fused_Mul_1009310095_const" type="Const" version="opset1">
			<data element_type="f32" offset="423196" shape="288,48,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="409" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="data_add_93229327/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="478492" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="Add1_8210/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="411" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="1434914352_const" type="Const" version="opset1">
			<data element_type="f32" offset="479644" shape="288,1,1,3,3" size="10368"/>
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
		<layer id="144" name="412" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="data_add_93309335/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="490012" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="Add1_8054/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="414" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="Mul1_8341/Fused_Mul_1010110103_const" type="Const" version="opset1">
			<data element_type="f32" offset="491164" shape="48,288,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="415" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="data_add_93389343/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="546460" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="Add1_8342/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="417" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="Mul1_8377/Fused_Mul_1010510107_const" type="Const" version="opset1">
			<data element_type="f32" offset="546652" shape="288,48,1,1" size="55296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="418" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>288</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="data_add_93469351/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="601948" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="Add1_8378/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="420" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="1431314316_const" type="Const" version="opset1">
			<data element_type="f32" offset="603100" shape="288,1,1,3,3" size="10368"/>
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
		<layer id="159" name="421" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="160" name="data_add_93549359/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="613468" shape="1,288,1,1" size="1152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="Add1_8330/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="162" name="423" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="Mul1_8065/Fused_Mul_1011310115_const" type="Const" version="opset1">
			<data element_type="f32" offset="614620" shape="72,288,1,1" size="82944"/>
			<output>
				<port id="1" precision="FP32">
					<dim>72</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="424" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>72</dim>
					<dim>288</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="data_add_93629367/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="697564" shape="1,72,1,1" size="288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="Add1_8066/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="Mul1_8281/Fused_Mul_1011710119_const" type="Const" version="opset1">
			<data element_type="f32" offset="697852" shape="432,72,1,1" size="124416"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="426" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="data_add_93709375/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="822268" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="Add1_8282/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="428" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="1427714280_const" type="Const" version="opset1">
			<data element_type="f32" offset="823996" shape="432,1,1,3,3" size="15552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="173" name="429" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="data_add_93789383/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="839548" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="Add1_7934/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="431" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="177" name="Mul1_8449/Fused_Mul_1012510127_const" type="Const" version="opset1">
			<data element_type="f32" offset="841276" shape="72,432,1,1" size="124416"/>
			<output>
				<port id="1" precision="FP32">
					<dim>72</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="432" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>72</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="179" name="data_add_93869391/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="965692" shape="1,72,1,1" size="288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="Add1_8450/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="181" name="434" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="Mul1_8077/Fused_Mul_1012910131_const" type="Const" version="opset1">
			<data element_type="f32" offset="965980" shape="432,72,1,1" size="124416"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="435" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="data_add_93949399/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1090396" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="Add1_8078/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="437" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="1433314336_const" type="Const" version="opset1">
			<data element_type="f32" offset="1092124" shape="432,1,1,3,3" size="15552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="188" name="438" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="data_add_94029407/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1107676" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="Add1_8114/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="440" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="192" name="Mul1_7909/Fused_Mul_1013710139_const" type="Const" version="opset1">
			<data element_type="f32" offset="1109404" shape="72,432,1,1" size="124416"/>
			<output>
				<port id="1" precision="FP32">
					<dim>72</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="441" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>72</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="194" name="data_add_94109415/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1233820" shape="1,72,1,1" size="288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="Add1_7910/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="443" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="Mul1_7921/Fused_Mul_1014110143_const" type="Const" version="opset1">
			<data element_type="f32" offset="1234108" shape="432,72,1,1" size="124416"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="444" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>72</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="data_add_94189423/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1358524" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="Add1_7922/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="446" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="1432514328_const" type="Const" version="opset1">
			<data element_type="f32" offset="1360252" shape="432,1,1,3,3" size="15552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="203" name="447" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="data_add_94269431/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1375804" shape="1,432,1,1" size="1728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="Add1_7982/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="449" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="207" name="Mul1_8497/Fused_Mul_1014910151_const" type="Const" version="opset1">
			<data element_type="f32" offset="1377532" shape="120,432,1,1" size="207360"/>
			<output>
				<port id="1" precision="FP32">
					<dim>120</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="450" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>432</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>120</dim>
					<dim>432</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="data_add_94349439/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1584892" shape="1,120,1,1" size="480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="Add1_8498/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="Mul1_8485/Fused_Mul_1015310155_const" type="Const" version="opset1">
			<data element_type="f32" offset="1585372" shape="720,120,1,1" size="345600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>720</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="212" name="452" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>720</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="data_add_94429447/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1930972" shape="1,720,1,1" size="2880"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="Add1_8486/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="215" name="454" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="216" name="1432114324_const" type="Const" version="opset1">
			<data element_type="f32" offset="1933852" shape="720,1,1,3,3" size="25920"/>
			<output>
				<port id="1" precision="FP32">
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="455" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="data_add_94509455/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1959772" shape="1,720,1,1" size="2880"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="219" name="Add1_8102/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="457" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="221" name="Mul1_8413/Fused_Mul_1016110163_const" type="Const" version="opset1">
			<data element_type="f32" offset="1962652" shape="120,720,1,1" size="345600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>120</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="458" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>120</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="data_add_94589463/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2308252" shape="1,120,1,1" size="480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="Add1_8414/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="460" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="226" name="Mul1_8389/Fused_Mul_1016510167_const" type="Const" version="opset1">
			<data element_type="f32" offset="2308732" shape="720,120,1,1" size="345600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>720</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="461" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>720</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="data_add_94669471/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2654332" shape="1,720,1,1" size="2880"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="Add1_8390/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="463" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="1432914332_const" type="Const" version="opset1">
			<data element_type="f32" offset="2657212" shape="720,1,1,3,3" size="25920"/>
			<output>
				<port id="1" precision="FP32">
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="464" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="data_add_94749479/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2683132" shape="1,720,1,1" size="2880"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="Add1_8462/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="466" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="Mul1_8521/Fused_Mul_1017310175_const" type="Const" version="opset1">
			<data element_type="f32" offset="2686012" shape="120,720,1,1" size="345600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>120</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="237" name="467" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>720</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>120</dim>
					<dim>720</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="238" name="data_add_94829487/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3031612" shape="1,120,1,1" size="480"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="Add1_8522/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
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
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="240" name="469" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="241" name="Mul1_8161/Fused_Mul_1017710179_const" type="Const" version="opset1">
			<data element_type="f32" offset="3032092" shape="480,120,1,1" size="230400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>480</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="470" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>480</dim>
					<dim>120</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="243" name="data_add_94909495/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3262492" shape="1,480,1,1" size="1920"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="Add1_8162/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="245" name="472" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="246" name="1434114344_const" type="Const" version="opset1">
			<data element_type="f32" offset="3264412" shape="480,1,1,3,3" size="17280"/>
			<output>
				<port id="1" precision="FP32">
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="473" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="248" name="data_add_94989503/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3281692" shape="1,480,1,1" size="1920"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="Add1_8030/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="250" name="475" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="251" name="Mul1_7945/Fused_Mul_1018510187_const" type="Const" version="opset1">
			<data element_type="f32" offset="3283612" shape="360,480,1,1" size="691200"/>
			<output>
				<port id="1" precision="FP32">
					<dim>360</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="252" name="476" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>480</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>360</dim>
					<dim>480</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="253" name="data_add_95069511/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3974812" shape="1,360,1,1" size="1440"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="Add1_7946/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="255" name="1430114304_const" type="Const" version="opset1">
			<data element_type="f32" offset="3976252" shape="360,1,1,3,3" size="12960"/>
			<output>
				<port id="1" precision="FP32">
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="256" name="482/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="data_add_95149519/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3989212" shape="1,360,1,1" size="1440"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="258" name="482/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="484" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="260" name="onnx_initializer_node_bbox_head.reg_convs.0.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="3990652" shape="36,360,1,1" size="51840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="261" name="485/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="485/Dims5628/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4042492" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="263" name="485" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="264" name="517/Cast_114622_const" type="Const" version="opset1">
			<data element_type="i64" offset="4042636" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="265" name="517" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="266" name="525/Cast_114614_const" type="Const" version="opset1">
			<data element_type="i64" offset="4042668" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="267" name="525" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12996</dim>
				</port>
			</output>
		</layer>
		<layer id="268" name="1429714300_const" type="Const" version="opset1">
			<data element_type="f32" offset="4042684" shape="360,1,1,3,3" size="12960"/>
			<output>
				<port id="1" precision="FP32">
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="478/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="270" name="data_add_95229527/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4055644" shape="1,360,1,1" size="1440"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="271" name="478/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="480" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="273" name="onnx_initializer_node_bbox_head.cls_convs.0.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4057084" shape="18,360,1,1" size="25920"/>
			<output>
				<port id="1" precision="FP32">
					<dim>18</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="481/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>18</dim>
					<dim>360</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="275" name="481/Dims5610/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4083004" shape="1,18,1,1" size="72"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="481" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>18</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="488/Cast_114612_const" type="Const" version="opset1">
			<data element_type="i64" offset="4042636" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="488" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>18</dim>
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
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="496/Cast_114610_const" type="Const" version="opset1">
			<data element_type="i64" offset="4042668" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="496" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>19</dim>
					<dim>19</dim>
					<dim>18</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6498</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="507/Cast_114602_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083076" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="507" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6498</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3249</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="508" type="SoftMax" version="opset1">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3249</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3249</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="516/Cast_114600_const" type="Const" version="opset1">
			<data element_type="i64" offset="4042668" shape="2" size="16"/>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="516" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3249</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6498</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="486/0_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>360</dim>
					<dim>19</dim>
					<dim>19</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="15257" type="Convert" version="opset1">
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
		<layer id="288" name="486/ss_0_port/Cast_114604_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083100" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="289" name="486/ss_0_port/Cast_214606_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083108" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="290" name="486/ss_0_port/Cast_314608_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083116" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="291" name="486/ss_0_port" type="StridedSlice" version="opset1">
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
		<layer id="292" name="15253" type="Convert" version="opset1">
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
		<layer id="293" name="486/1_port" type="ShapeOf" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>300</dim>
					<dim>300</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="294" name="15259" type="Convert" version="opset1">
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
		<layer id="295" name="486/ss_1_port/Cast_114616_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083100" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="486/ss_1_port/Cast_214618_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083108" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="297" name="486/ss_1_port/Cast_314620_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083116" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="298" name="486/ss_1_port" type="StridedSlice" version="opset1">
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
		<layer id="299" name="15255" type="Convert" version="opset1">
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
		<layer id="300" name="486/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="15.0,39.6,25.5,63.2,227.5,162.9,124.5,105.1,72.6" img_h="0" img_size="0" img_w="0" offset="0.5" step="16.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="9.4,25.1,14.7,34.7,143.0,77.4,128.8,51.1,75.6"/>
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
					<dim>12996</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="486/unsqueeze/value12116_const" type="Const" version="opset1">
			<data element_type="i64" offset="4083124" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="486" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>12996</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>12996</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="527" type="DetectionOutput" version="opset1">
			<data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.019999999552965164" eta="1.0" height="0" height_scale="0" input_height="1" input_width="1" interp_mode="" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="1" num_classes="2" pad_mode="" pad_value="" prob="0" resize_mode="" share_location="1" top_k="200" variance_encoded_in_target="0" visualize_threshold="0.6" width="0" width_scale="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12996</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>6498</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>12996</dim>
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
		<layer id="304" name="527/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="8" from-port="2" to-layer="10" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="1"/>
		<edge from-layer="10" from-port="2" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="13" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="13" to-port="1"/>
		<edge from-layer="13" from-port="2" to-layer="15" to-port="0"/>
		<edge from-layer="14" from-port="1" to-layer="15" to-port="1"/>
		<edge from-layer="15" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="1"/>
		<edge from-layer="18" from-port="2" to-layer="20" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="20" to-port="1"/>
		<edge from-layer="20" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="24" to-port="0"/>
		<edge from-layer="23" from-port="1" to-layer="24" to-port="1"/>
		<edge from-layer="24" from-port="2" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="27" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="27" to-port="1"/>
		<edge from-layer="27" from-port="2" to-layer="29" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="1"/>
		<edge from-layer="29" from-port="2" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="34" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="1"/>
		<edge from-layer="34" from-port="2" to-layer="36" to-port="0"/>
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
		<edge from-layer="34" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="48" from-port="2" to-layer="49" to-port="1"/>
		<edge from-layer="49" from-port="2" to-layer="51" to-port="0"/>
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
		<edge from-layer="63" from-port="2" to-layer="65" to-port="0"/>
		<edge from-layer="64" from-port="1" to-layer="65" to-port="1"/>
		<edge from-layer="65" from-port="2" to-layer="67" to-port="0"/>
		<edge from-layer="66" from-port="1" to-layer="67" to-port="1"/>
		<edge from-layer="67" from-port="2" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="1" to-layer="70" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="72" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="72" to-port="1"/>
		<edge from-layer="72" from-port="2" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="75" to-port="0"/>
		<edge from-layer="74" from-port="1" to-layer="75" to-port="1"/>
		<edge from-layer="75" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="76" from-port="1" to-layer="77" to-port="1"/>
		<edge from-layer="63" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="77" from-port="2" to-layer="78" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="80" to-port="0"/>
		<edge from-layer="79" from-port="1" to-layer="80" to-port="1"/>
		<edge from-layer="80" from-port="2" to-layer="82" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="82" to-port="1"/>
		<edge from-layer="82" from-port="2" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="1" to-layer="85" to-port="0"/>
		<edge from-layer="84" from-port="1" to-layer="85" to-port="1"/>
		<edge from-layer="85" from-port="2" to-layer="87" to-port="0"/>
		<edge from-layer="86" from-port="1" to-layer="87" to-port="1"/>
		<edge from-layer="87" from-port="2" to-layer="88" to-port="0"/>
		<edge from-layer="88" from-port="1" to-layer="90" to-port="0"/>
		<edge from-layer="89" from-port="1" to-layer="90" to-port="1"/>
		<edge from-layer="90" from-port="2" to-layer="92" to-port="0"/>
		<edge from-layer="91" from-port="1" to-layer="92" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="93" to-port="0"/>
		<edge from-layer="92" from-port="2" to-layer="93" to-port="1"/>
		<edge from-layer="93" from-port="2" to-layer="95" to-port="0"/>
		<edge from-layer="94" from-port="1" to-layer="95" to-port="1"/>
		<edge from-layer="95" from-port="2" to-layer="97" to-port="0"/>
		<edge from-layer="96" from-port="1" to-layer="97" to-port="1"/>
		<edge from-layer="97" from-port="2" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="100" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="100" to-port="1"/>
		<edge from-layer="100" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="101" from-port="1" to-layer="102" to-port="1"/>
		<edge from-layer="102" from-port="2" to-layer="103" to-port="0"/>
		<edge from-layer="103" from-port="1" to-layer="105" to-port="0"/>
		<edge from-layer="104" from-port="1" to-layer="105" to-port="1"/>
		<edge from-layer="105" from-port="2" to-layer="107" to-port="0"/>
		<edge from-layer="106" from-port="1" to-layer="107" to-port="1"/>
		<edge from-layer="107" from-port="2" to-layer="109" to-port="0"/>
		<edge from-layer="108" from-port="1" to-layer="109" to-port="1"/>
		<edge from-layer="109" from-port="2" to-layer="111" to-port="0"/>
		<edge from-layer="110" from-port="1" to-layer="111" to-port="1"/>
		<edge from-layer="111" from-port="2" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="114" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="114" to-port="1"/>
		<edge from-layer="114" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="116" to-port="1"/>
		<edge from-layer="116" from-port="2" to-layer="117" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="119" to-port="0"/>
		<edge from-layer="118" from-port="1" to-layer="119" to-port="1"/>
		<edge from-layer="119" from-port="2" to-layer="121" to-port="0"/>
		<edge from-layer="120" from-port="1" to-layer="121" to-port="1"/>
		<edge from-layer="107" from-port="2" to-layer="122" to-port="0"/>
		<edge from-layer="121" from-port="2" to-layer="122" to-port="1"/>
		<edge from-layer="122" from-port="2" to-layer="124" to-port="0"/>
		<edge from-layer="123" from-port="1" to-layer="124" to-port="1"/>
		<edge from-layer="124" from-port="2" to-layer="126" to-port="0"/>
		<edge from-layer="125" from-port="1" to-layer="126" to-port="1"/>
		<edge from-layer="126" from-port="2" to-layer="127" to-port="0"/>
		<edge from-layer="127" from-port="1" to-layer="129" to-port="0"/>
		<edge from-layer="128" from-port="1" to-layer="129" to-port="1"/>
		<edge from-layer="129" from-port="2" to-layer="131" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="131" to-port="1"/>
		<edge from-layer="131" from-port="2" to-layer="132" to-port="0"/>
		<edge from-layer="132" from-port="1" to-layer="134" to-port="0"/>
		<edge from-layer="133" from-port="1" to-layer="134" to-port="1"/>
		<edge from-layer="134" from-port="2" to-layer="136" to-port="0"/>
		<edge from-layer="135" from-port="1" to-layer="136" to-port="1"/>
		<edge from-layer="122" from-port="2" to-layer="137" to-port="0"/>
		<edge from-layer="136" from-port="2" to-layer="137" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="139" to-port="0"/>
		<edge from-layer="138" from-port="1" to-layer="139" to-port="1"/>
		<edge from-layer="139" from-port="2" to-layer="141" to-port="0"/>
		<edge from-layer="140" from-port="1" to-layer="141" to-port="1"/>
		<edge from-layer="141" from-port="2" to-layer="142" to-port="0"/>
		<edge from-layer="142" from-port="1" to-layer="144" to-port="0"/>
		<edge from-layer="143" from-port="1" to-layer="144" to-port="1"/>
		<edge from-layer="144" from-port="2" to-layer="146" to-port="0"/>
		<edge from-layer="145" from-port="1" to-layer="146" to-port="1"/>
		<edge from-layer="146" from-port="2" to-layer="147" to-port="0"/>
		<edge from-layer="147" from-port="1" to-layer="149" to-port="0"/>
		<edge from-layer="148" from-port="1" to-layer="149" to-port="1"/>
		<edge from-layer="149" from-port="2" to-layer="151" to-port="0"/>
		<edge from-layer="150" from-port="1" to-layer="151" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="152" to-port="0"/>
		<edge from-layer="151" from-port="2" to-layer="152" to-port="1"/>
		<edge from-layer="152" from-port="2" to-layer="154" to-port="0"/>
		<edge from-layer="153" from-port="1" to-layer="154" to-port="1"/>
		<edge from-layer="154" from-port="2" to-layer="156" to-port="0"/>
		<edge from-layer="155" from-port="1" to-layer="156" to-port="1"/>
		<edge from-layer="156" from-port="2" to-layer="157" to-port="0"/>
		<edge from-layer="157" from-port="1" to-layer="159" to-port="0"/>
		<edge from-layer="158" from-port="1" to-layer="159" to-port="1"/>
		<edge from-layer="159" from-port="2" to-layer="161" to-port="0"/>
		<edge from-layer="160" from-port="1" to-layer="161" to-port="1"/>
		<edge from-layer="161" from-port="2" to-layer="162" to-port="0"/>
		<edge from-layer="162" from-port="1" to-layer="164" to-port="0"/>
		<edge from-layer="163" from-port="1" to-layer="164" to-port="1"/>
		<edge from-layer="164" from-port="2" to-layer="166" to-port="0"/>
		<edge from-layer="165" from-port="1" to-layer="166" to-port="1"/>
		<edge from-layer="166" from-port="2" to-layer="168" to-port="0"/>
		<edge from-layer="167" from-port="1" to-layer="168" to-port="1"/>
		<edge from-layer="168" from-port="2" to-layer="170" to-port="0"/>
		<edge from-layer="169" from-port="1" to-layer="170" to-port="1"/>
		<edge from-layer="170" from-port="2" to-layer="171" to-port="0"/>
		<edge from-layer="171" from-port="1" to-layer="173" to-port="0"/>
		<edge from-layer="172" from-port="1" to-layer="173" to-port="1"/>
		<edge from-layer="173" from-port="2" to-layer="175" to-port="0"/>
		<edge from-layer="174" from-port="1" to-layer="175" to-port="1"/>
		<edge from-layer="175" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="176" from-port="1" to-layer="178" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="178" to-port="1"/>
		<edge from-layer="178" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="179" from-port="1" to-layer="180" to-port="1"/>
		<edge from-layer="166" from-port="2" to-layer="181" to-port="0"/>
		<edge from-layer="180" from-port="2" to-layer="181" to-port="1"/>
		<edge from-layer="181" from-port="2" to-layer="183" to-port="0"/>
		<edge from-layer="182" from-port="1" to-layer="183" to-port="1"/>
		<edge from-layer="183" from-port="2" to-layer="185" to-port="0"/>
		<edge from-layer="184" from-port="1" to-layer="185" to-port="1"/>
		<edge from-layer="185" from-port="2" to-layer="186" to-port="0"/>
		<edge from-layer="186" from-port="1" to-layer="188" to-port="0"/>
		<edge from-layer="187" from-port="1" to-layer="188" to-port="1"/>
		<edge from-layer="188" from-port="2" to-layer="190" to-port="0"/>
		<edge from-layer="189" from-port="1" to-layer="190" to-port="1"/>
		<edge from-layer="190" from-port="2" to-layer="191" to-port="0"/>
		<edge from-layer="191" from-port="1" to-layer="193" to-port="0"/>
		<edge from-layer="192" from-port="1" to-layer="193" to-port="1"/>
		<edge from-layer="193" from-port="2" to-layer="195" to-port="0"/>
		<edge from-layer="194" from-port="1" to-layer="195" to-port="1"/>
		<edge from-layer="181" from-port="2" to-layer="196" to-port="0"/>
		<edge from-layer="195" from-port="2" to-layer="196" to-port="1"/>
		<edge from-layer="196" from-port="2" to-layer="198" to-port="0"/>
		<edge from-layer="197" from-port="1" to-layer="198" to-port="1"/>
		<edge from-layer="198" from-port="2" to-layer="200" to-port="0"/>
		<edge from-layer="199" from-port="1" to-layer="200" to-port="1"/>
		<edge from-layer="200" from-port="2" to-layer="201" to-port="0"/>
		<edge from-layer="201" from-port="1" to-layer="203" to-port="0"/>
		<edge from-layer="202" from-port="1" to-layer="203" to-port="1"/>
		<edge from-layer="203" from-port="2" to-layer="205" to-port="0"/>
		<edge from-layer="204" from-port="1" to-layer="205" to-port="1"/>
		<edge from-layer="205" from-port="2" to-layer="206" to-port="0"/>
		<edge from-layer="206" from-port="1" to-layer="208" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="208" to-port="1"/>
		<edge from-layer="208" from-port="2" to-layer="210" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="210" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="212" to-port="0"/>
		<edge from-layer="211" from-port="1" to-layer="212" to-port="1"/>
		<edge from-layer="212" from-port="2" to-layer="214" to-port="0"/>
		<edge from-layer="213" from-port="1" to-layer="214" to-port="1"/>
		<edge from-layer="214" from-port="2" to-layer="215" to-port="0"/>
		<edge from-layer="215" from-port="1" to-layer="217" to-port="0"/>
		<edge from-layer="216" from-port="1" to-layer="217" to-port="1"/>
		<edge from-layer="217" from-port="2" to-layer="219" to-port="0"/>
		<edge from-layer="218" from-port="1" to-layer="219" to-port="1"/>
		<edge from-layer="219" from-port="2" to-layer="220" to-port="0"/>
		<edge from-layer="220" from-port="1" to-layer="222" to-port="0"/>
		<edge from-layer="221" from-port="1" to-layer="222" to-port="1"/>
		<edge from-layer="222" from-port="2" to-layer="224" to-port="0"/>
		<edge from-layer="223" from-port="1" to-layer="224" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="225" to-port="0"/>
		<edge from-layer="224" from-port="2" to-layer="225" to-port="1"/>
		<edge from-layer="225" from-port="2" to-layer="227" to-port="0"/>
		<edge from-layer="226" from-port="1" to-layer="227" to-port="1"/>
		<edge from-layer="227" from-port="2" to-layer="229" to-port="0"/>
		<edge from-layer="228" from-port="1" to-layer="229" to-port="1"/>
		<edge from-layer="229" from-port="2" to-layer="230" to-port="0"/>
		<edge from-layer="230" from-port="1" to-layer="232" to-port="0"/>
		<edge from-layer="231" from-port="1" to-layer="232" to-port="1"/>
		<edge from-layer="232" from-port="2" to-layer="234" to-port="0"/>
		<edge from-layer="233" from-port="1" to-layer="234" to-port="1"/>
		<edge from-layer="234" from-port="2" to-layer="235" to-port="0"/>
		<edge from-layer="235" from-port="1" to-layer="237" to-port="0"/>
		<edge from-layer="236" from-port="1" to-layer="237" to-port="1"/>
		<edge from-layer="237" from-port="2" to-layer="239" to-port="0"/>
		<edge from-layer="238" from-port="1" to-layer="239" to-port="1"/>
		<edge from-layer="225" from-port="2" to-layer="240" to-port="0"/>
		<edge from-layer="239" from-port="2" to-layer="240" to-port="1"/>
		<edge from-layer="240" from-port="2" to-layer="242" to-port="0"/>
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
		<edge from-layer="254" from-port="2" to-layer="256" to-port="0"/>
		<edge from-layer="255" from-port="1" to-layer="256" to-port="1"/>
		<edge from-layer="256" from-port="2" to-layer="258" to-port="0"/>
		<edge from-layer="257" from-port="1" to-layer="258" to-port="1"/>
		<edge from-layer="258" from-port="2" to-layer="259" to-port="0"/>
		<edge from-layer="259" from-port="1" to-layer="261" to-port="0"/>
		<edge from-layer="260" from-port="1" to-layer="261" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="263" to-port="0"/>
		<edge from-layer="262" from-port="1" to-layer="263" to-port="1"/>
		<edge from-layer="263" from-port="2" to-layer="265" to-port="0"/>
		<edge from-layer="264" from-port="1" to-layer="265" to-port="1"/>
		<edge from-layer="265" from-port="2" to-layer="267" to-port="0"/>
		<edge from-layer="266" from-port="1" to-layer="267" to-port="1"/>
		<edge from-layer="254" from-port="2" to-layer="269" to-port="0"/>
		<edge from-layer="268" from-port="1" to-layer="269" to-port="1"/>
		<edge from-layer="269" from-port="2" to-layer="271" to-port="0"/>
		<edge from-layer="270" from-port="1" to-layer="271" to-port="1"/>
		<edge from-layer="271" from-port="2" to-layer="272" to-port="0"/>
		<edge from-layer="272" from-port="1" to-layer="274" to-port="0"/>
		<edge from-layer="273" from-port="1" to-layer="274" to-port="1"/>
		<edge from-layer="274" from-port="2" to-layer="276" to-port="0"/>
		<edge from-layer="275" from-port="1" to-layer="276" to-port="1"/>
		<edge from-layer="276" from-port="2" to-layer="278" to-port="0"/>
		<edge from-layer="277" from-port="1" to-layer="278" to-port="1"/>
		<edge from-layer="278" from-port="2" to-layer="280" to-port="0"/>
		<edge from-layer="279" from-port="1" to-layer="280" to-port="1"/>
		<edge from-layer="280" from-port="2" to-layer="282" to-port="0"/>
		<edge from-layer="281" from-port="1" to-layer="282" to-port="1"/>
		<edge from-layer="282" from-port="2" to-layer="283" to-port="0"/>
		<edge from-layer="283" from-port="1" to-layer="285" to-port="0"/>
		<edge from-layer="284" from-port="1" to-layer="285" to-port="1"/>
		<edge from-layer="254" from-port="2" to-layer="286" to-port="0"/>
		<edge from-layer="286" from-port="1" to-layer="287" to-port="0"/>
		<edge from-layer="287" from-port="1" to-layer="291" to-port="0"/>
		<edge from-layer="288" from-port="1" to-layer="291" to-port="1"/>
		<edge from-layer="289" from-port="1" to-layer="291" to-port="2"/>
		<edge from-layer="290" from-port="1" to-layer="291" to-port="3"/>
		<edge from-layer="291" from-port="4" to-layer="292" to-port="0"/>
		<edge from-layer="2" from-port="2" to-layer="293" to-port="0"/>
		<edge from-layer="293" from-port="1" to-layer="294" to-port="0"/>
		<edge from-layer="294" from-port="1" to-layer="298" to-port="0"/>
		<edge from-layer="295" from-port="1" to-layer="298" to-port="1"/>
		<edge from-layer="296" from-port="1" to-layer="298" to-port="2"/>
		<edge from-layer="297" from-port="1" to-layer="298" to-port="3"/>
		<edge from-layer="298" from-port="4" to-layer="299" to-port="0"/>
		<edge from-layer="292" from-port="1" to-layer="300" to-port="0"/>
		<edge from-layer="299" from-port="1" to-layer="300" to-port="1"/>
		<edge from-layer="300" from-port="2" to-layer="302" to-port="0"/>
		<edge from-layer="301" from-port="1" to-layer="302" to-port="1"/>
		<edge from-layer="267" from-port="2" to-layer="303" to-port="0"/>
		<edge from-layer="285" from-port="2" to-layer="303" to-port="1"/>
		<edge from-layer="302" from-port="2" to-layer="303" to-port="2"/>
		<edge from-layer="303" from-port="3" to-layer="304" to-port="0"/>
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
			<framework value="onnx"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V2 value="False"/>
			<generate_deprecated_IR_V7 value="False"/>
			<generate_experimental_IR_V10 value="True"/>
			<input value="input.1"/>
			<input_model value="DIR/mobilenetv2_fd_single_head.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,300,300]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'input.1': {'mean': None, 'scale': array([254.99991075])}}"/>
			<mean_values value="()"/>
			<model_name value="face-detection-retail-0005"/>
			<move_to_preprocess value="False"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'input.1': array([  1,   3, 300, 300])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input.1[254.99991075003123]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, output, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
