<?xml version="1.0" ?>
<net name="person-vehicle-bike-detection-2002" version="10">
	<layers>
		<layer id="0" name="image" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,512,512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
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
					<dim>512</dim>
					<dim>512</dim>
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
					<dim>512</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="image/reverse_input_channels/Concat20938_const" type="Const" version="opset1">
			<data element_type="f32" offset="4" shape="32,3,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Conv_0" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="data_add_12292/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3460" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="BatchNormalization_1/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Clip_2" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="BatchNormalization_4/mean/Fused_Mul_1341613418_const" type="Const" version="opset1">
			<data element_type="f32" offset="3588" shape="32,32,1,1" size="4096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Conv_3" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="data_add_1229512300/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7684" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="BatchNormalization_4/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="Clip_5" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="2163321636_const" type="Const" version="opset1">
			<data element_type="f32" offset="7812" shape="32,1,1,3,3" size="1152"/>
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
		<layer id="14" name="Conv_6" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="data_add_1230312308/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8964" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="BatchNormalization_7/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="Clip_8" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="BatchNormalization_10/mean/Fused_Mul_1342413426_const" type="Const" version="opset1">
			<data element_type="f32" offset="9092" shape="16,32,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="Conv_9" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
				<port id="1">
					<dim>16</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="data_add_1231112316/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11140" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="BatchNormalization_10/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="BatchNormalization_12/mean/Fused_Mul_1342813430_const" type="Const" version="opset1">
			<data element_type="f32" offset="11204" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="Conv_11" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="data_add_1231912324/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="17348" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="BatchNormalization_12/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="Clip_13" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="2168121684_const" type="Const" version="opset1">
			<data element_type="f32" offset="17732" shape="96,1,1,3,3" size="3456"/>
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
		<layer id="28" name="Conv_14" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>256</dim>
					<dim>256</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="data_add_1232712332/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="21188" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="BatchNormalization_15/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="Clip_16" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="BatchNormalization_18/mean/Fused_Mul_1343613438_const" type="Const" version="opset1">
			<data element_type="f32" offset="21572" shape="24,96,1,1" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="Conv_17" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="data_add_1233512340/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="30788" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="BatchNormalization_18/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="BatchNormalization_20/mean/Fused_Mul_1344013442_const" type="Const" version="opset1">
			<data element_type="f32" offset="30884" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="Conv_19" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="data_add_1234312348/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="44708" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="BatchNormalization_20/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="Clip_21" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="2164521648_const" type="Const" version="opset1">
			<data element_type="f32" offset="45284" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="42" name="Conv_22" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="data_add_1235112356/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="50468" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="BatchNormalization_23/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="Clip_24" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="BatchNormalization_26/mean/Fused_Mul_1344813450_const" type="Const" version="opset1">
			<data element_type="f32" offset="51044" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="Conv_25" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="data_add_1235912364/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="64868" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="BatchNormalization_26/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="Add_27" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="BatchNormalization_29/mean/Fused_Mul_1345213454_const" type="Const" version="opset1">
			<data element_type="f32" offset="64964" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="Conv_28" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="data_add_1236712372/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="78788" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="BatchNormalization_29/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="Clip_30" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="2165721660_const" type="Const" version="opset1">
			<data element_type="f32" offset="79364" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="57" name="Conv_31" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>128</dim>
					<dim>128</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="data_add_1237512380/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="84548" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="BatchNormalization_32/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="Clip_33" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="BatchNormalization_35/mean/Fused_Mul_1346013462_const" type="Const" version="opset1">
			<data element_type="f32" offset="85124" shape="32,144,1,1" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="Conv_34" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="data_add_1238312388/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="103556" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="BatchNormalization_35/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="BatchNormalization_37/mean/Fused_Mul_1346413466_const" type="Const" version="opset1">
			<data element_type="f32" offset="103684" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="Conv_36" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="data_add_1239112396/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="128260" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="BatchNormalization_37/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="Clip_38" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="2164121644_const" type="Const" version="opset1">
			<data element_type="f32" offset="129028" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="71" name="Conv_39" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="data_add_1239912404/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="135940" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="BatchNormalization_40/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="Clip_41" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="BatchNormalization_43/mean/Fused_Mul_1347213474_const" type="Const" version="opset1">
			<data element_type="f32" offset="136708" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="Conv_42" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="data_add_1240712412/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="161284" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="BatchNormalization_43/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="Add_44" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="BatchNormalization_46/mean/Fused_Mul_1347613478_const" type="Const" version="opset1">
			<data element_type="f32" offset="161412" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="Conv_45" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="data_add_1241512420/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="185988" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="BatchNormalization_46/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="Clip_47" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="2166121664_const" type="Const" version="opset1">
			<data element_type="f32" offset="186756" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="86" name="Conv_48" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="data_add_1242312428/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="193668" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="BatchNormalization_49/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="Clip_50" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="BatchNormalization_52/mean/Fused_Mul_1348413486_const" type="Const" version="opset1">
			<data element_type="f32" offset="194436" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="Conv_51" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="data_add_1243112436/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="219012" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="BatchNormalization_52/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="Add_53" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="BatchNormalization_55/mean/Fused_Mul_1348813490_const" type="Const" version="opset1">
			<data element_type="f32" offset="219140" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="Conv_54" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="data_add_1243912444/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="243716" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="BatchNormalization_55/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="Clip_56" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="2166921672_const" type="Const" version="opset1">
			<data element_type="f32" offset="244484" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="101" name="Conv_57" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>64</dim>
					<dim>64</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="data_add_1244712452/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="251396" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="BatchNormalization_58/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="Clip_59" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="BatchNormalization_61/mean/Fused_Mul_1349613498_const" type="Const" version="opset1">
			<data element_type="f32" offset="252164" shape="64,192,1,1" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="Conv_60" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="data_add_1245512460/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="301316" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="BatchNormalization_61/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="BatchNormalization_63/mean/Fused_Mul_1350013502_const" type="Const" version="opset1">
			<data element_type="f32" offset="301572" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="Conv_62" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="data_add_1246312468/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="399876" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="BatchNormalization_63/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="Clip_64" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="2168521688_const" type="Const" version="opset1">
			<data element_type="f32" offset="401412" shape="384,1,1,3,3" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="Conv_65" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="data_add_1247112476/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="415236" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="BatchNormalization_66/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="Clip_67" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="BatchNormalization_69/mean/Fused_Mul_1350813510_const" type="Const" version="opset1">
			<data element_type="f32" offset="416772" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="Conv_68" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="data_add_1247912484/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="515076" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="BatchNormalization_69/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="Add_70" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="BatchNormalization_72/mean/Fused_Mul_1351213514_const" type="Const" version="opset1">
			<data element_type="f32" offset="515332" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="Conv_71" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="data_add_1248712492/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="613636" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="BatchNormalization_72/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="Clip_73" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="2169721700_const" type="Const" version="opset1">
			<data element_type="f32" offset="615172" shape="384,1,1,3,3" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="Conv_74" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="data_add_1249512500/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="628996" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="BatchNormalization_75/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="Clip_76" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="BatchNormalization_78/mean/Fused_Mul_1352013522_const" type="Const" version="opset1">
			<data element_type="f32" offset="630532" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="Conv_77" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="data_add_1250312508/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="728836" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="BatchNormalization_78/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="Add_79" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="BatchNormalization_81/mean/Fused_Mul_1352413526_const" type="Const" version="opset1">
			<data element_type="f32" offset="729092" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="Conv_80" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="data_add_1251112516/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="827396" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="BatchNormalization_81/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="Clip_82" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="2167721680_const" type="Const" version="opset1">
			<data element_type="f32" offset="828932" shape="384,1,1,3,3" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="Conv_83" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="data_add_1251912524/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="842756" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="BatchNormalization_84/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="Clip_85" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="BatchNormalization_87/mean/Fused_Mul_1353213534_const" type="Const" version="opset1">
			<data element_type="f32" offset="844292" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="Conv_86" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="data_add_1252712532/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="942596" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="BatchNormalization_87/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="Add_88" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="BatchNormalization_90/mean/Fused_Mul_1353613538_const" type="Const" version="opset1">
			<data element_type="f32" offset="942852" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="Conv_89" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="data_add_1253512540/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1041156" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="BatchNormalization_90/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="Clip_91" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="159" name="2165321656_const" type="Const" version="opset1">
			<data element_type="f32" offset="1042692" shape="384,1,1,3,3" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="160" name="Conv_92" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="data_add_1254312548/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1056516" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="162" name="BatchNormalization_93/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="Clip_94" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="BatchNormalization_96/mean/Fused_Mul_1354413546_const" type="Const" version="opset1">
			<data element_type="f32" offset="1058052" shape="96,384,1,1" size="147456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="Conv_95" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="data_add_1255112556/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1205508" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="BatchNormalization_96/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="BatchNormalization_98/mean/Fused_Mul_1354813550_const" type="Const" version="opset1">
			<data element_type="f32" offset="1205892" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="Conv_97" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="data_add_1255912564/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1427076" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="BatchNormalization_98/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="Clip_99" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="173" name="2163721640_const" type="Const" version="opset1">
			<data element_type="f32" offset="1429380" shape="576,1,1,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="Conv_100" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="data_add_1256712572/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1450116" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="BatchNormalization_101/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="177" name="Clip_102" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="BatchNormalization_104/mean/Fused_Mul_1355613558_const" type="Const" version="opset1">
			<data element_type="f32" offset="1452420" shape="96,576,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="179" name="Conv_103" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="data_add_1257512580/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1673604" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="181" name="BatchNormalization_104/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="Add_105" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="BatchNormalization_107/mean/Fused_Mul_1356013562_const" type="Const" version="opset1">
			<data element_type="f32" offset="1673988" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="Conv_106" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="data_add_1258312588/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1895172" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="BatchNormalization_107/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="Clip_108" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="188" name="2169321696_const" type="Const" version="opset1">
			<data element_type="f32" offset="1897476" shape="576,1,1,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="Conv_109" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="data_add_1259112596/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1918212" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="BatchNormalization_110/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="192" name="Clip_111" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="BatchNormalization_113/mean/Fused_Mul_1356813570_const" type="Const" version="opset1">
			<data element_type="f32" offset="1920516" shape="96,576,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="194" name="Conv_112" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="data_add_1259912604/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2141700" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="BatchNormalization_113/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="Add_114" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="2168921692_const" type="Const" version="opset1">
			<data element_type="f32" offset="2142084" shape="96,1,1,3,3" size="3456"/>
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
		<layer id="199" name="Conv_153/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="data_add_1260712612/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2145540" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="Conv_153/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="Relu_155" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="203" name="onnx_initializer_node_bbox_head.reg_convs.0.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2145924" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="Conv_156/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="Conv_156/Dims7227/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2152068" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="Conv_156" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="207" name="Transpose_196/Cast_122114_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152132" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="Transpose_196" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="Shape_197" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="Gather_199547/Cast_122042_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="211" name="Gather_199547/Cast_222044_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="212" name="Gather_199547" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="213" name="Unsqueeze_200/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="Unsqueeze_200/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="215" name="onnx_initializer_node_581/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="216" name="Concat_201" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="Reshape_202" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16384</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="BatchNormalization_116/mean/Fused_Mul_1358013582_const" type="Const" version="opset1">
			<data element_type="f32" offset="2152184" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="219" name="Conv_115" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="data_add_1262312628/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2373368" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="221" name="BatchNormalization_116/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="Clip_117" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="2170521708_const" type="Const" version="opset1">
			<data element_type="f32" offset="2375672" shape="576,1,1,3,3" size="20736"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="Conv_118" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="data_add_1263112636/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2396408" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="226" name="BatchNormalization_119/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="Clip_120" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="BatchNormalization_122/mean/Fused_Mul_1358813590_const" type="Const" version="opset1">
			<data element_type="f32" offset="2398712" shape="160,576,1,1" size="368640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="Conv_121" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>160</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="data_add_1263912644/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2767352" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="BatchNormalization_122/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="BatchNormalization_124/mean/Fused_Mul_1359213594_const" type="Const" version="opset1">
			<data element_type="f32" offset="2767992" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="Conv_123" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="data_add_1264712652/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3382392" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="BatchNormalization_124/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="Clip_125" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="237" name="2162921632_const" type="Const" version="opset1">
			<data element_type="f32" offset="3386232" shape="960,1,1,3,3" size="34560"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="238" name="Conv_126" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="data_add_1265512660/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3420792" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="240" name="BatchNormalization_127/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="241" name="Clip_128" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="BatchNormalization_130/mean/Fused_Mul_1360013602_const" type="Const" version="opset1">
			<data element_type="f32" offset="3424632" shape="160,960,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="243" name="Conv_129" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="data_add_1266312668/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4039032" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="245" name="BatchNormalization_130/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="246" name="Add_131" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="BatchNormalization_133/mean/Fused_Mul_1360413606_const" type="Const" version="opset1">
			<data element_type="f32" offset="4039672" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="248" name="Conv_132" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="data_add_1267112676/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4654072" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="250" name="BatchNormalization_133/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="251" name="Clip_134" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="252" name="2170121704_const" type="Const" version="opset1">
			<data element_type="f32" offset="4657912" shape="960,1,1,3,3" size="34560"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="253" name="Conv_135" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="data_add_1267912684/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4692472" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="255" name="BatchNormalization_136/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="256" name="Clip_137" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="BatchNormalization_139/mean/Fused_Mul_1361213614_const" type="Const" version="opset1">
			<data element_type="f32" offset="4696312" shape="160,960,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="258" name="Conv_138" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="data_add_1268712692/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5310712" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="260" name="BatchNormalization_139/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="261" name="Add_140" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="BatchNormalization_142/mean/Fused_Mul_1361613618_const" type="Const" version="opset1">
			<data element_type="f32" offset="5311352" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="263" name="Conv_141" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>160</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="264" name="data_add_1269512700/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5925752" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="265" name="BatchNormalization_142/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="266" name="Clip_143" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="267" name="2166521668_const" type="Const" version="opset1">
			<data element_type="f32" offset="5929592" shape="960,1,1,3,3" size="34560"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="268" name="Conv_144" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="data_add_1270312708/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5964152" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="270" name="BatchNormalization_145/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="271" name="Clip_146" type="Clamp" version="opset1">
			<data max="6.0" min="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="BatchNormalization_148/mean/Fused_Mul_1362413626_const" type="Const" version="opset1">
			<data element_type="f32" offset="5967992" shape="320,960,1,1" size="1228800"/>
			<output>
				<port id="1" precision="FP32">
					<dim>320</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="273" name="Conv_147" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>960</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>320</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="data_add_1271112716/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7196792" shape="1,320,1,1" size="1280"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="275" name="BatchNormalization_148/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="2167321676_const" type="Const" version="opset1">
			<data element_type="f32" offset="7198072" shape="320,1,1,3,3" size="11520"/>
			<output>
				<port id="1" precision="FP32">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="Conv_161/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="data_add_1271912724/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7209592" shape="1,320,1,1" size="1280"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="Conv_161/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="Relu_163" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="onnx_initializer_node_bbox_head.reg_convs.1.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="7210872" shape="20,320,1,1" size="25600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="Conv_164/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="Conv_164/Dims7233/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7236472" shape="1,20,1,1" size="80"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="Conv_164" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="Transpose_203/Cast_122066_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152132" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="Transpose_203" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="Shape_204" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="288" name="Gather_206549/Cast_122100_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="289" name="Gather_206549/Cast_222102_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="290" name="Gather_206549" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="291" name="Unsqueeze_207/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="292" name="Unsqueeze_207/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="293" name="onnx_initializer_node_582/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="294" name="Concat_208" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="295" name="Reshape_209" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5120</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="Concat_210" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16384</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>5120</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
			</output>
		</layer>
		<layer id="297" name="2170921712_const" type="Const" version="opset1">
			<data element_type="f32" offset="7236552" shape="96,1,1,3,3" size="3456"/>
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
		<layer id="298" name="Conv_149/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="299" name="data_add_1261512620/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7240008" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="300" name="Conv_149/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="Relu_151" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="onnx_initializer_node_bbox_head.cls_convs.0.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="7240392" shape="16,96,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="Conv_152/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="304" name="Conv_152/Dims7215/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7246536" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="305" name="Conv_152" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="306" name="Transpose_168/Cast_122038_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152132" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="307" name="Transpose_168" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="308" name="Shape_169" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="309" name="Gather_171553/Cast_122104_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="310" name="Gather_171553/Cast_222106_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="311" name="Gather_171553" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="312" name="Unsqueeze_172/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="313" name="Unsqueeze_172/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="314" name="onnx_initializer_node_576/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="315" name="Concat_173" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="316" name="Reshape_174" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16384</dim>
				</port>
			</output>
		</layer>
		<layer id="317" name="2164921652_const" type="Const" version="opset1">
			<data element_type="f32" offset="7246600" shape="320,1,1,3,3" size="11520"/>
			<output>
				<port id="1" precision="FP32">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="318" name="Conv_157/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="319" name="data_add_1272712732/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7258120" shape="1,320,1,1" size="1280"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="320" name="Conv_157/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="321" name="Relu_159" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="322" name="onnx_initializer_node_bbox_head.cls_convs.1.3.weight/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="7259400" shape="20,320,1,1" size="25600"/>
			<output>
				<port id="1" precision="FP32">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="323" name="Conv_160/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="324" name="Conv_160/Dims7251/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7285000" shape="1,20,1,1" size="80"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="325" name="Conv_160" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="326" name="Transpose_175/Cast_122068_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152132" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="327" name="Transpose_175" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="328" name="Shape_176" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>20</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="329" name="Gather_178545/Cast_122046_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="330" name="Gather_178545/Cast_222048_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="331" name="Gather_178545" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>4</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="332" name="Unsqueeze_179/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="333" name="Unsqueeze_179/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="334" name="onnx_initializer_node_577/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="335" name="Concat_180" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="336" name="Reshape_181" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5120</dim>
				</port>
			</output>
		</layer>
		<layer id="337" name="Concat_182" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16384</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>5120</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
			</output>
		</layer>
		<layer id="338" name="Shape_183" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="339" name="Gather_185555/Cast_122076_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="340" name="Gather_185555/Cast_222078_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="341" name="Gather_185555" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="342" name="Unsqueeze_186/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="343" name="Unsqueeze_186/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="344" name="onnx_initializer_node_578/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="345" name="onnx_initializer_node_579/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="7285080" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="346" name="Concat_187" type="Concat" version="opset1">
			<data axis="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="347" name="Reshape_188" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="348" name="Softmax_189/FlattenONNX_/input_shape" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="349" name="Softmax_189/FlattenONNX_/input_shape/Gather/Cast_122050_const" type="Const" version="opset1">
			<data element_type="i32" offset="7285088" shape="2" size="8"/>
			<output>
				<port id="1" precision="I32">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="350" name="Softmax_189/FlattenONNX_/input_shape/Gather/Cast_222052_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="351" name="Softmax_189/FlattenONNX_/input_shape/Gather" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="352" name="Softmax_189/FlattenONNX_/first_dims/Cast_122040_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="353" name="Softmax_189/FlattenONNX_/first_dims" type="ReduceProd" version="opset1">
			<data keep_dims="True"/>
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="354" name="Softmax_189/FlattenONNX_/second_dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="355" name="Softmax_189/FlattenONNX_/first_dims/shapes_concat" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="356" name="Softmax_189/FlattenONNX_/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="357" name="Softmax_189/Softmax_" type="SoftMax" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="358" name="Softmax_189/ShapeOf_" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="359" name="Softmax_189" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>5376</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="360" name="Shape_190" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="361" name="Gather_192551/Cast_122080_const" type="Const" version="opset1">
			<data element_type="i32" offset="2152164" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="362" name="Gather_192551/Cast_222082_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="363" name="Gather_192551" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64"/>
			</output>
		</layer>
		<layer id="364" name="Unsqueeze_193/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="365" name="Unsqueeze_193/Unsqueeze" type="Unsqueeze" version="opset1">
			<input>
				<port id="0"/>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="366" name="onnx_initializer_node_580/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2152176" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="367" name="Concat_194" type="Concat" version="opset1">
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
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="368" name="Reshape_195" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5376</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
			</output>
		</layer>
		<layer id="369" name="PriorBoxClustered_165/0_port" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="370" name="PriorBoxClustered_165/ss_0_port/Cast_122090_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285096" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="371" name="PriorBoxClustered_165/ss_0_port/Cast_222092_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285080" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="372" name="PriorBoxClustered_165/ss_0_port/Cast_322094_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285104" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="373" name="PriorBoxClustered_165/ss_0_port" type="StridedSlice" version="opset1">
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
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="374" name="PriorBoxClustered_165/1_port" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="375" name="PriorBoxClustered_165/ss_1_port/Cast_122054_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285096" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="376" name="PriorBoxClustered_165/ss_1_port/Cast_222056_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285080" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="377" name="PriorBoxClustered_165/ss_1_port/Cast_322058_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285104" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="378" name="PriorBoxClustered_165/ss_1_port" type="StridedSlice" version="opset1">
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
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="379" name="PriorBoxClustered_165/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="24.572557,58.396797,107.00043,170.78336" img_h="0" img_size="0" img_w="0" offset="0.5" step="16.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="14.328363,26.013193,30.63051,47.144855"/>
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
					<dim>16384</dim>
				</port>
			</output>
		</layer>
		<layer id="380" name="PriorBoxClustered_165/unsqueeze/value16357_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="381" name="PriorBoxClustered_165" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>16384</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16384</dim>
				</port>
			</output>
		</layer>
		<layer id="382" name="PriorBoxClustered_166/0_port" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="383" name="PriorBoxClustered_166/ss_0_port/Cast_122060_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285096" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="384" name="PriorBoxClustered_166/ss_0_port/Cast_222062_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285080" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="385" name="PriorBoxClustered_166/ss_0_port/Cast_322064_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285104" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="386" name="PriorBoxClustered_166/ss_0_port" type="StridedSlice" version="opset1">
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
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="387" name="PriorBoxClustered_166/1_port" type="ShapeOf" version="opset1">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="388" name="PriorBoxClustered_166/ss_1_port/Cast_122108_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285096" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="389" name="PriorBoxClustered_166/ss_1_port/Cast_222110_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285080" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="390" name="PriorBoxClustered_166/ss_1_port/Cast_322112_const" type="Const" version="opset1">
			<data element_type="i64" offset="7285104" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="391" name="PriorBoxClustered_166/ss_1_port" type="StridedSlice" version="opset1">
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
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="392" name="PriorBoxClustered_166/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="85.73938,253.05121,380.506,210.23965,292.85803" img_h="0" img_size="0" img_w="0" offset="0.5" step="32.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="103.72894,65.18865,99.059944,183.24388,300.66544"/>
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
					<dim>5120</dim>
				</port>
			</output>
		</layer>
		<layer id="393" name="PriorBoxClustered_166/unsqueeze/value16339_const" type="Const" version="opset1">
			<data element_type="i64" offset="2152168" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="394" name="PriorBoxClustered_166" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>5120</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>5120</dim>
				</port>
			</output>
		</layer>
		<layer id="395" name="Concat_167" type="Concat" version="opset1">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16384</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>5120</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>21504</dim>
				</port>
			</output>
		</layer>
		<layer id="396" name="detection_out" type="DetectionOutput" version="opset1">
			<data background_label_id="3" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.019999999552965164" eta="1.0" height="0" height_scale="0" input_height="1" input_width="1" interp_mode="" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="1" num_classes="4" pad_mode="" pad_value="" prob="0" resize_mode="" share_location="1" top_k="200" variance_encoded_in_target="0" visualize_threshold="0.6" width="0" width_scale="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>21504</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>21504</dim>
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
		<layer id="397" name="detection_out/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="9" to-port="1"/>
		<edge from-layer="9" from-port="2" to-layer="11" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="11" to-port="1"/>
		<edge from-layer="11" from-port="2" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="14" to-port="1"/>
		<edge from-layer="14" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="1"/>
		<edge from-layer="16" from-port="2" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="19" to-port="0"/>
		<edge from-layer="18" from-port="1" to-layer="19" to-port="1"/>
		<edge from-layer="19" from-port="2" to-layer="21" to-port="0"/>
		<edge from-layer="20" from-port="1" to-layer="21" to-port="1"/>
		<edge from-layer="21" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="1"/>
		<edge from-layer="23" from-port="2" to-layer="25" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="1"/>
		<edge from-layer="25" from-port="2" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="28" to-port="0"/>
		<edge from-layer="27" from-port="1" to-layer="28" to-port="1"/>
		<edge from-layer="28" from-port="2" to-layer="30" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="1"/>
		<edge from-layer="30" from-port="2" to-layer="31" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="33" to-port="0"/>
		<edge from-layer="32" from-port="1" to-layer="33" to-port="1"/>
		<edge from-layer="33" from-port="2" to-layer="35" to-port="0"/>
		<edge from-layer="34" from-port="1" to-layer="35" to-port="1"/>
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="1" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="39" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="39" to-port="1"/>
		<edge from-layer="39" from-port="2" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="44" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="44" to-port="1"/>
		<edge from-layer="44" from-port="2" to-layer="45" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="1" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="1"/>
		<edge from-layer="49" from-port="2" to-layer="50" to-port="0"/>
		<edge from-layer="35" from-port="2" to-layer="50" to-port="1"/>
		<edge from-layer="50" from-port="2" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="1" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="54" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="54" to-port="1"/>
		<edge from-layer="54" from-port="2" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="57" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="57" to-port="1"/>
		<edge from-layer="57" from-port="2" to-layer="59" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="1"/>
		<edge from-layer="59" from-port="2" to-layer="60" to-port="0"/>
		<edge from-layer="60" from-port="1" to-layer="62" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="62" to-port="1"/>
		<edge from-layer="62" from-port="2" to-layer="64" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="1"/>
		<edge from-layer="64" from-port="2" to-layer="66" to-port="0"/>
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
		<edge from-layer="64" from-port="2" to-layer="79" to-port="1"/>
		<edge from-layer="79" from-port="2" to-layer="81" to-port="0"/>
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
		<edge from-layer="79" from-port="2" to-layer="94" to-port="1"/>
		<edge from-layer="94" from-port="2" to-layer="96" to-port="0"/>
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
		<edge from-layer="108" from-port="2" to-layer="110" to-port="0"/>
		<edge from-layer="109" from-port="1" to-layer="110" to-port="1"/>
		<edge from-layer="110" from-port="2" to-layer="112" to-port="0"/>
		<edge from-layer="111" from-port="1" to-layer="112" to-port="1"/>
		<edge from-layer="112" from-port="2" to-layer="113" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="115" to-port="0"/>
		<edge from-layer="114" from-port="1" to-layer="115" to-port="1"/>
		<edge from-layer="115" from-port="2" to-layer="117" to-port="0"/>
		<edge from-layer="116" from-port="1" to-layer="117" to-port="1"/>
		<edge from-layer="117" from-port="2" to-layer="118" to-port="0"/>
		<edge from-layer="118" from-port="1" to-layer="120" to-port="0"/>
		<edge from-layer="119" from-port="1" to-layer="120" to-port="1"/>
		<edge from-layer="120" from-port="2" to-layer="122" to-port="0"/>
		<edge from-layer="121" from-port="1" to-layer="122" to-port="1"/>
		<edge from-layer="122" from-port="2" to-layer="123" to-port="0"/>
		<edge from-layer="108" from-port="2" to-layer="123" to-port="1"/>
		<edge from-layer="123" from-port="2" to-layer="125" to-port="0"/>
		<edge from-layer="124" from-port="1" to-layer="125" to-port="1"/>
		<edge from-layer="125" from-port="2" to-layer="127" to-port="0"/>
		<edge from-layer="126" from-port="1" to-layer="127" to-port="1"/>
		<edge from-layer="127" from-port="2" to-layer="128" to-port="0"/>
		<edge from-layer="128" from-port="1" to-layer="130" to-port="0"/>
		<edge from-layer="129" from-port="1" to-layer="130" to-port="1"/>
		<edge from-layer="130" from-port="2" to-layer="132" to-port="0"/>
		<edge from-layer="131" from-port="1" to-layer="132" to-port="1"/>
		<edge from-layer="132" from-port="2" to-layer="133" to-port="0"/>
		<edge from-layer="133" from-port="1" to-layer="135" to-port="0"/>
		<edge from-layer="134" from-port="1" to-layer="135" to-port="1"/>
		<edge from-layer="135" from-port="2" to-layer="137" to-port="0"/>
		<edge from-layer="136" from-port="1" to-layer="137" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="138" to-port="0"/>
		<edge from-layer="123" from-port="2" to-layer="138" to-port="1"/>
		<edge from-layer="138" from-port="2" to-layer="140" to-port="0"/>
		<edge from-layer="139" from-port="1" to-layer="140" to-port="1"/>
		<edge from-layer="140" from-port="2" to-layer="142" to-port="0"/>
		<edge from-layer="141" from-port="1" to-layer="142" to-port="1"/>
		<edge from-layer="142" from-port="2" to-layer="143" to-port="0"/>
		<edge from-layer="143" from-port="1" to-layer="145" to-port="0"/>
		<edge from-layer="144" from-port="1" to-layer="145" to-port="1"/>
		<edge from-layer="145" from-port="2" to-layer="147" to-port="0"/>
		<edge from-layer="146" from-port="1" to-layer="147" to-port="1"/>
		<edge from-layer="147" from-port="2" to-layer="148" to-port="0"/>
		<edge from-layer="148" from-port="1" to-layer="150" to-port="0"/>
		<edge from-layer="149" from-port="1" to-layer="150" to-port="1"/>
		<edge from-layer="150" from-port="2" to-layer="152" to-port="0"/>
		<edge from-layer="151" from-port="1" to-layer="152" to-port="1"/>
		<edge from-layer="152" from-port="2" to-layer="153" to-port="0"/>
		<edge from-layer="138" from-port="2" to-layer="153" to-port="1"/>
		<edge from-layer="153" from-port="2" to-layer="155" to-port="0"/>
		<edge from-layer="154" from-port="1" to-layer="155" to-port="1"/>
		<edge from-layer="155" from-port="2" to-layer="157" to-port="0"/>
		<edge from-layer="156" from-port="1" to-layer="157" to-port="1"/>
		<edge from-layer="157" from-port="2" to-layer="158" to-port="0"/>
		<edge from-layer="158" from-port="1" to-layer="160" to-port="0"/>
		<edge from-layer="159" from-port="1" to-layer="160" to-port="1"/>
		<edge from-layer="160" from-port="2" to-layer="162" to-port="0"/>
		<edge from-layer="161" from-port="1" to-layer="162" to-port="1"/>
		<edge from-layer="162" from-port="2" to-layer="163" to-port="0"/>
		<edge from-layer="163" from-port="1" to-layer="165" to-port="0"/>
		<edge from-layer="164" from-port="1" to-layer="165" to-port="1"/>
		<edge from-layer="165" from-port="2" to-layer="167" to-port="0"/>
		<edge from-layer="166" from-port="1" to-layer="167" to-port="1"/>
		<edge from-layer="167" from-port="2" to-layer="169" to-port="0"/>
		<edge from-layer="168" from-port="1" to-layer="169" to-port="1"/>
		<edge from-layer="169" from-port="2" to-layer="171" to-port="0"/>
		<edge from-layer="170" from-port="1" to-layer="171" to-port="1"/>
		<edge from-layer="171" from-port="2" to-layer="172" to-port="0"/>
		<edge from-layer="172" from-port="1" to-layer="174" to-port="0"/>
		<edge from-layer="173" from-port="1" to-layer="174" to-port="1"/>
		<edge from-layer="174" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="175" from-port="1" to-layer="176" to-port="1"/>
		<edge from-layer="176" from-port="2" to-layer="177" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="179" to-port="0"/>
		<edge from-layer="178" from-port="1" to-layer="179" to-port="1"/>
		<edge from-layer="179" from-port="2" to-layer="181" to-port="0"/>
		<edge from-layer="180" from-port="1" to-layer="181" to-port="1"/>
		<edge from-layer="181" from-port="2" to-layer="182" to-port="0"/>
		<edge from-layer="167" from-port="2" to-layer="182" to-port="1"/>
		<edge from-layer="182" from-port="2" to-layer="184" to-port="0"/>
		<edge from-layer="183" from-port="1" to-layer="184" to-port="1"/>
		<edge from-layer="184" from-port="2" to-layer="186" to-port="0"/>
		<edge from-layer="185" from-port="1" to-layer="186" to-port="1"/>
		<edge from-layer="186" from-port="2" to-layer="187" to-port="0"/>
		<edge from-layer="187" from-port="1" to-layer="189" to-port="0"/>
		<edge from-layer="188" from-port="1" to-layer="189" to-port="1"/>
		<edge from-layer="189" from-port="2" to-layer="191" to-port="0"/>
		<edge from-layer="190" from-port="1" to-layer="191" to-port="1"/>
		<edge from-layer="191" from-port="2" to-layer="192" to-port="0"/>
		<edge from-layer="192" from-port="1" to-layer="194" to-port="0"/>
		<edge from-layer="193" from-port="1" to-layer="194" to-port="1"/>
		<edge from-layer="194" from-port="2" to-layer="196" to-port="0"/>
		<edge from-layer="195" from-port="1" to-layer="196" to-port="1"/>
		<edge from-layer="196" from-port="2" to-layer="197" to-port="0"/>
		<edge from-layer="182" from-port="2" to-layer="197" to-port="1"/>
		<edge from-layer="197" from-port="2" to-layer="199" to-port="0"/>
		<edge from-layer="198" from-port="1" to-layer="199" to-port="1"/>
		<edge from-layer="199" from-port="2" to-layer="201" to-port="0"/>
		<edge from-layer="200" from-port="1" to-layer="201" to-port="1"/>
		<edge from-layer="201" from-port="2" to-layer="202" to-port="0"/>
		<edge from-layer="202" from-port="1" to-layer="204" to-port="0"/>
		<edge from-layer="203" from-port="1" to-layer="204" to-port="1"/>
		<edge from-layer="204" from-port="2" to-layer="206" to-port="0"/>
		<edge from-layer="205" from-port="1" to-layer="206" to-port="1"/>
		<edge from-layer="206" from-port="2" to-layer="208" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="208" to-port="1"/>
		<edge from-layer="206" from-port="2" to-layer="209" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="212" to-port="0"/>
		<edge from-layer="210" from-port="1" to-layer="212" to-port="1"/>
		<edge from-layer="211" from-port="1" to-layer="212" to-port="2"/>
		<edge from-layer="212" from-port="3" to-layer="214" to-port="0"/>
		<edge from-layer="213" from-port="1" to-layer="214" to-port="1"/>
		<edge from-layer="214" from-port="2" to-layer="216" to-port="0"/>
		<edge from-layer="215" from-port="1" to-layer="216" to-port="1"/>
		<edge from-layer="208" from-port="2" to-layer="217" to-port="0"/>
		<edge from-layer="216" from-port="2" to-layer="217" to-port="1"/>
		<edge from-layer="197" from-port="2" to-layer="219" to-port="0"/>
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
		<edge from-layer="231" from-port="2" to-layer="233" to-port="0"/>
		<edge from-layer="232" from-port="1" to-layer="233" to-port="1"/>
		<edge from-layer="233" from-port="2" to-layer="235" to-port="0"/>
		<edge from-layer="234" from-port="1" to-layer="235" to-port="1"/>
		<edge from-layer="235" from-port="2" to-layer="236" to-port="0"/>
		<edge from-layer="236" from-port="1" to-layer="238" to-port="0"/>
		<edge from-layer="237" from-port="1" to-layer="238" to-port="1"/>
		<edge from-layer="238" from-port="2" to-layer="240" to-port="0"/>
		<edge from-layer="239" from-port="1" to-layer="240" to-port="1"/>
		<edge from-layer="240" from-port="2" to-layer="241" to-port="0"/>
		<edge from-layer="241" from-port="1" to-layer="243" to-port="0"/>
		<edge from-layer="242" from-port="1" to-layer="243" to-port="1"/>
		<edge from-layer="243" from-port="2" to-layer="245" to-port="0"/>
		<edge from-layer="244" from-port="1" to-layer="245" to-port="1"/>
		<edge from-layer="245" from-port="2" to-layer="246" to-port="0"/>
		<edge from-layer="231" from-port="2" to-layer="246" to-port="1"/>
		<edge from-layer="246" from-port="2" to-layer="248" to-port="0"/>
		<edge from-layer="247" from-port="1" to-layer="248" to-port="1"/>
		<edge from-layer="248" from-port="2" to-layer="250" to-port="0"/>
		<edge from-layer="249" from-port="1" to-layer="250" to-port="1"/>
		<edge from-layer="250" from-port="2" to-layer="251" to-port="0"/>
		<edge from-layer="251" from-port="1" to-layer="253" to-port="0"/>
		<edge from-layer="252" from-port="1" to-layer="253" to-port="1"/>
		<edge from-layer="253" from-port="2" to-layer="255" to-port="0"/>
		<edge from-layer="254" from-port="1" to-layer="255" to-port="1"/>
		<edge from-layer="255" from-port="2" to-layer="256" to-port="0"/>
		<edge from-layer="256" from-port="1" to-layer="258" to-port="0"/>
		<edge from-layer="257" from-port="1" to-layer="258" to-port="1"/>
		<edge from-layer="258" from-port="2" to-layer="260" to-port="0"/>
		<edge from-layer="259" from-port="1" to-layer="260" to-port="1"/>
		<edge from-layer="260" from-port="2" to-layer="261" to-port="0"/>
		<edge from-layer="246" from-port="2" to-layer="261" to-port="1"/>
		<edge from-layer="261" from-port="2" to-layer="263" to-port="0"/>
		<edge from-layer="262" from-port="1" to-layer="263" to-port="1"/>
		<edge from-layer="263" from-port="2" to-layer="265" to-port="0"/>
		<edge from-layer="264" from-port="1" to-layer="265" to-port="1"/>
		<edge from-layer="265" from-port="2" to-layer="266" to-port="0"/>
		<edge from-layer="266" from-port="1" to-layer="268" to-port="0"/>
		<edge from-layer="267" from-port="1" to-layer="268" to-port="1"/>
		<edge from-layer="268" from-port="2" to-layer="270" to-port="0"/>
		<edge from-layer="269" from-port="1" to-layer="270" to-port="1"/>
		<edge from-layer="270" from-port="2" to-layer="271" to-port="0"/>
		<edge from-layer="271" from-port="1" to-layer="273" to-port="0"/>
		<edge from-layer="272" from-port="1" to-layer="273" to-port="1"/>
		<edge from-layer="273" from-port="2" to-layer="275" to-port="0"/>
		<edge from-layer="274" from-port="1" to-layer="275" to-port="1"/>
		<edge from-layer="275" from-port="2" to-layer="277" to-port="0"/>
		<edge from-layer="276" from-port="1" to-layer="277" to-port="1"/>
		<edge from-layer="277" from-port="2" to-layer="279" to-port="0"/>
		<edge from-layer="278" from-port="1" to-layer="279" to-port="1"/>
		<edge from-layer="279" from-port="2" to-layer="280" to-port="0"/>
		<edge from-layer="280" from-port="1" to-layer="282" to-port="0"/>
		<edge from-layer="281" from-port="1" to-layer="282" to-port="1"/>
		<edge from-layer="282" from-port="2" to-layer="284" to-port="0"/>
		<edge from-layer="283" from-port="1" to-layer="284" to-port="1"/>
		<edge from-layer="284" from-port="2" to-layer="286" to-port="0"/>
		<edge from-layer="285" from-port="1" to-layer="286" to-port="1"/>
		<edge from-layer="284" from-port="2" to-layer="287" to-port="0"/>
		<edge from-layer="287" from-port="1" to-layer="290" to-port="0"/>
		<edge from-layer="288" from-port="1" to-layer="290" to-port="1"/>
		<edge from-layer="289" from-port="1" to-layer="290" to-port="2"/>
		<edge from-layer="290" from-port="3" to-layer="292" to-port="0"/>
		<edge from-layer="291" from-port="1" to-layer="292" to-port="1"/>
		<edge from-layer="292" from-port="2" to-layer="294" to-port="0"/>
		<edge from-layer="293" from-port="1" to-layer="294" to-port="1"/>
		<edge from-layer="286" from-port="2" to-layer="295" to-port="0"/>
		<edge from-layer="294" from-port="2" to-layer="295" to-port="1"/>
		<edge from-layer="217" from-port="2" to-layer="296" to-port="0"/>
		<edge from-layer="295" from-port="2" to-layer="296" to-port="1"/>
		<edge from-layer="197" from-port="2" to-layer="298" to-port="0"/>
		<edge from-layer="297" from-port="1" to-layer="298" to-port="1"/>
		<edge from-layer="298" from-port="2" to-layer="300" to-port="0"/>
		<edge from-layer="299" from-port="1" to-layer="300" to-port="1"/>
		<edge from-layer="300" from-port="2" to-layer="301" to-port="0"/>
		<edge from-layer="301" from-port="1" to-layer="303" to-port="0"/>
		<edge from-layer="302" from-port="1" to-layer="303" to-port="1"/>
		<edge from-layer="303" from-port="2" to-layer="305" to-port="0"/>
		<edge from-layer="304" from-port="1" to-layer="305" to-port="1"/>
		<edge from-layer="305" from-port="2" to-layer="307" to-port="0"/>
		<edge from-layer="306" from-port="1" to-layer="307" to-port="1"/>
		<edge from-layer="305" from-port="2" to-layer="308" to-port="0"/>
		<edge from-layer="308" from-port="1" to-layer="311" to-port="0"/>
		<edge from-layer="309" from-port="1" to-layer="311" to-port="1"/>
		<edge from-layer="310" from-port="1" to-layer="311" to-port="2"/>
		<edge from-layer="311" from-port="3" to-layer="313" to-port="0"/>
		<edge from-layer="312" from-port="1" to-layer="313" to-port="1"/>
		<edge from-layer="313" from-port="2" to-layer="315" to-port="0"/>
		<edge from-layer="314" from-port="1" to-layer="315" to-port="1"/>
		<edge from-layer="307" from-port="2" to-layer="316" to-port="0"/>
		<edge from-layer="315" from-port="2" to-layer="316" to-port="1"/>
		<edge from-layer="275" from-port="2" to-layer="318" to-port="0"/>
		<edge from-layer="317" from-port="1" to-layer="318" to-port="1"/>
		<edge from-layer="318" from-port="2" to-layer="320" to-port="0"/>
		<edge from-layer="319" from-port="1" to-layer="320" to-port="1"/>
		<edge from-layer="320" from-port="2" to-layer="321" to-port="0"/>
		<edge from-layer="321" from-port="1" to-layer="323" to-port="0"/>
		<edge from-layer="322" from-port="1" to-layer="323" to-port="1"/>
		<edge from-layer="323" from-port="2" to-layer="325" to-port="0"/>
		<edge from-layer="324" from-port="1" to-layer="325" to-port="1"/>
		<edge from-layer="325" from-port="2" to-layer="327" to-port="0"/>
		<edge from-layer="326" from-port="1" to-layer="327" to-port="1"/>
		<edge from-layer="325" from-port="2" to-layer="328" to-port="0"/>
		<edge from-layer="328" from-port="1" to-layer="331" to-port="0"/>
		<edge from-layer="329" from-port="1" to-layer="331" to-port="1"/>
		<edge from-layer="330" from-port="1" to-layer="331" to-port="2"/>
		<edge from-layer="331" from-port="3" to-layer="333" to-port="0"/>
		<edge from-layer="332" from-port="1" to-layer="333" to-port="1"/>
		<edge from-layer="333" from-port="2" to-layer="335" to-port="0"/>
		<edge from-layer="334" from-port="1" to-layer="335" to-port="1"/>
		<edge from-layer="327" from-port="2" to-layer="336" to-port="0"/>
		<edge from-layer="335" from-port="2" to-layer="336" to-port="1"/>
		<edge from-layer="316" from-port="2" to-layer="337" to-port="0"/>
		<edge from-layer="336" from-port="2" to-layer="337" to-port="1"/>
		<edge from-layer="337" from-port="2" to-layer="338" to-port="0"/>
		<edge from-layer="338" from-port="1" to-layer="341" to-port="0"/>
		<edge from-layer="339" from-port="1" to-layer="341" to-port="1"/>
		<edge from-layer="340" from-port="1" to-layer="341" to-port="2"/>
		<edge from-layer="341" from-port="3" to-layer="343" to-port="0"/>
		<edge from-layer="342" from-port="1" to-layer="343" to-port="1"/>
		<edge from-layer="343" from-port="2" to-layer="346" to-port="0"/>
		<edge from-layer="344" from-port="1" to-layer="346" to-port="1"/>
		<edge from-layer="345" from-port="1" to-layer="346" to-port="2"/>
		<edge from-layer="337" from-port="2" to-layer="347" to-port="0"/>
		<edge from-layer="346" from-port="3" to-layer="347" to-port="1"/>
		<edge from-layer="347" from-port="2" to-layer="348" to-port="0"/>
		<edge from-layer="348" from-port="1" to-layer="351" to-port="0"/>
		<edge from-layer="349" from-port="1" to-layer="351" to-port="1"/>
		<edge from-layer="350" from-port="1" to-layer="351" to-port="2"/>
		<edge from-layer="351" from-port="3" to-layer="353" to-port="0"/>
		<edge from-layer="352" from-port="1" to-layer="353" to-port="1"/>
		<edge from-layer="353" from-port="2" to-layer="355" to-port="0"/>
		<edge from-layer="354" from-port="1" to-layer="355" to-port="1"/>
		<edge from-layer="347" from-port="2" to-layer="356" to-port="0"/>
		<edge from-layer="355" from-port="2" to-layer="356" to-port="1"/>
		<edge from-layer="356" from-port="2" to-layer="357" to-port="0"/>
		<edge from-layer="347" from-port="2" to-layer="358" to-port="0"/>
		<edge from-layer="357" from-port="1" to-layer="359" to-port="0"/>
		<edge from-layer="358" from-port="1" to-layer="359" to-port="1"/>
		<edge from-layer="359" from-port="2" to-layer="360" to-port="0"/>
		<edge from-layer="360" from-port="1" to-layer="363" to-port="0"/>
		<edge from-layer="361" from-port="1" to-layer="363" to-port="1"/>
		<edge from-layer="362" from-port="1" to-layer="363" to-port="2"/>
		<edge from-layer="363" from-port="3" to-layer="365" to-port="0"/>
		<edge from-layer="364" from-port="1" to-layer="365" to-port="1"/>
		<edge from-layer="365" from-port="2" to-layer="367" to-port="0"/>
		<edge from-layer="366" from-port="1" to-layer="367" to-port="1"/>
		<edge from-layer="359" from-port="2" to-layer="368" to-port="0"/>
		<edge from-layer="367" from-port="2" to-layer="368" to-port="1"/>
		<edge from-layer="197" from-port="2" to-layer="369" to-port="0"/>
		<edge from-layer="369" from-port="1" to-layer="373" to-port="0"/>
		<edge from-layer="370" from-port="1" to-layer="373" to-port="1"/>
		<edge from-layer="371" from-port="1" to-layer="373" to-port="2"/>
		<edge from-layer="372" from-port="1" to-layer="373" to-port="3"/>
		<edge from-layer="2" from-port="2" to-layer="374" to-port="0"/>
		<edge from-layer="374" from-port="1" to-layer="378" to-port="0"/>
		<edge from-layer="375" from-port="1" to-layer="378" to-port="1"/>
		<edge from-layer="376" from-port="1" to-layer="378" to-port="2"/>
		<edge from-layer="377" from-port="1" to-layer="378" to-port="3"/>
		<edge from-layer="373" from-port="4" to-layer="379" to-port="0"/>
		<edge from-layer="378" from-port="4" to-layer="379" to-port="1"/>
		<edge from-layer="379" from-port="2" to-layer="381" to-port="0"/>
		<edge from-layer="380" from-port="1" to-layer="381" to-port="1"/>
		<edge from-layer="275" from-port="2" to-layer="382" to-port="0"/>
		<edge from-layer="382" from-port="1" to-layer="386" to-port="0"/>
		<edge from-layer="383" from-port="1" to-layer="386" to-port="1"/>
		<edge from-layer="384" from-port="1" to-layer="386" to-port="2"/>
		<edge from-layer="385" from-port="1" to-layer="386" to-port="3"/>
		<edge from-layer="2" from-port="2" to-layer="387" to-port="0"/>
		<edge from-layer="387" from-port="1" to-layer="391" to-port="0"/>
		<edge from-layer="388" from-port="1" to-layer="391" to-port="1"/>
		<edge from-layer="389" from-port="1" to-layer="391" to-port="2"/>
		<edge from-layer="390" from-port="1" to-layer="391" to-port="3"/>
		<edge from-layer="386" from-port="4" to-layer="392" to-port="0"/>
		<edge from-layer="391" from-port="4" to-layer="392" to-port="1"/>
		<edge from-layer="392" from-port="2" to-layer="394" to-port="0"/>
		<edge from-layer="393" from-port="1" to-layer="394" to-port="1"/>
		<edge from-layer="381" from-port="2" to-layer="395" to-port="0"/>
		<edge from-layer="394" from-port="2" to-layer="395" to-port="1"/>
		<edge from-layer="296" from-port="2" to-layer="396" to-port="0"/>
		<edge from-layer="368" from-port="2" to-layer="396" to-port="1"/>
		<edge from-layer="395" from-port="2" to-layer="396" to-port="2"/>
		<edge from-layer="396" from-port="3" to-layer="397" to-port="0"/>
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
			<framework value="onnx"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input value="image"/>
			<input_model value="DIR/model.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,512,512]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'image': {'mean': None, 'scale': array([255.])}}"/>
			<mean_values value="()"/>
			<model_name value="person-vehicle-bike-detection-2002"/>
			<output value="['detection_out']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'image': array([  1,   3, 512, 512])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="image[255.0]"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
