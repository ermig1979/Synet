<?xml version="1.0" ?>
<net name="LPRNet" version="10">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,24,94"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data/reverse_input_channels/Concat2671_const" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="64,3,3,3" size="6912"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="conv1/Dims793/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="6912" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
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
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="relu_conv1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="pool1" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="51/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="7168" shape="32,64,1,1" size="8192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="fire1_small/conv_reduce/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="fire1_small/conv_reduce/Dims835/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="15360" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="fire1_small/conv_reduce" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
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
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="fire1_small/relu_conv_reduce" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="47/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="15488" shape="32,32,3,1" size="12288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="fire1_small/conv_3x1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>32</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="fire1_small/conv_3x1/Dims751/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="27776" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="fire1_small/conv_3x1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
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
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="fire1_small/relu_conv_3x1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="57/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="27904" shape="32,32,1,3" size="12288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="fire1_small/conv_1x3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
				<port id="1">
					<dim>32</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="fire1_small/conv_1x3/Dims823/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="40192" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="fire1_small/conv_1x3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
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
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="fire1_small/relu_conv_1x3" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="59/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="40320" shape="128,32,1,1" size="16384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="fire1_small/conv_expand/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
				<port id="1">
					<dim>128</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="fire1_small/conv_expand/Dims757/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="56704" shape="1,128,1,1" size="512"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="fire1_small/conv_expand" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
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
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="fire1_small/relu_conv_expand" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="pool2" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="77/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="57216" shape="64,128,1,1" size="32768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="fire2_small/conv_reduce/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="fire2_small/conv_reduce/Dims799/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="89984" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="fire2_small/conv_reduce" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="fire2_small/relu_conv_reduce" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="67/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="90240" shape="64,64,3,1" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="fire2_small/conv_3x1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="fire2_small/conv_3x1/Dims769/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="139392" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="fire2_small/conv_3x1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="fire2_small/relu_conv_3x1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="53/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="139648" shape="64,64,1,3" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="fire2_small/conv_1x3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="fire2_small/conv_1x3/Dims811/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="188800" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="fire2_small/conv_1x3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="fire2_small/relu_conv_1x3" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="63/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="189056" shape="256,64,1,1" size="65536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="fire2_small/conv_expand/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="fire2_small/conv_expand/Dims829/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="254592" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="fire2_small/conv_expand" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="fire2_small/relu_conv_expand" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="55/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="255616" shape="64,256,1,1" size="65536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="fire3_small/conv_reduce/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="fire3_small/conv_reduce/Dims817/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="321152" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="fire3_small/conv_reduce" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="fire3_small/relu_conv_reduce" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="65/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="321408" shape="64,64,3,1" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="fire3_small/conv_3x1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="fire3_small/conv_3x1/Dims841/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="370560" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="fire3_small/conv_3x1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="fire3_small/relu_conv_3x1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="49/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="370816" shape="64,64,1,3" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="fire3_small/conv_1x3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="fire3_small/conv_1x3/Dims805/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="419968" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="fire3_small/conv_1x3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
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
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="fire3_small/relu_conv_1x3" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="69/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="420224" shape="256,64,1,1" size="65536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="fire3_small/conv_expand/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="fire3_small/conv_expand/Dims787/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="485760" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="fire3_small/conv_expand" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="fire3_small/relu_conv_expand" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="pool3" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="45/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="486784" shape="128,256,5,1" size="655360"/>
			<output>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="conv2/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>128</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="conv2/Dims781/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1142144" shape="1,128,1,1" size="512"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="conv2" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
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
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="relu_conv2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="73/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1142656" shape="71,128,1,13" size="472576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>71</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>13</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="conv3_w/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,6" pads_end="0,6" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>71</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>13</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="conv3_w/Dims775/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1615232" shape="1,71,1,1" size="284"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="conv3_w" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="relu_conv3_w" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="pattern/weights_transpose/MinusOne1962_const" type="Const" version="opset1">
			<data element_type="i64" offset="1615516" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="pattern/WithoutBiases/1_port_transpose1913_const" type="Const" version="opset1">
			<data element_type="f32" offset="1615524" shape="128,6248" size="3198976"/>
			<output>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>6248</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="pattern/weights_transpose/Shape" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>128</dim>
					<dim>6248</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="pattern/weights_transpose/Shape/Gather/Cast_12925_const" type="Const" version="opset1">
			<data element_type="i32" offset="4814500" shape="1" size="4"/>
			<output>
				<port id="1" precision="I32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="pattern/weights_transpose/Shape/Gather/Cast_22927_const" type="Const" version="opset1">
			<data element_type="i64" offset="4814504" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="84" name="pattern/weights_transpose/Shape/Gather" type="Gather" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2"/>
			</input>
			<output>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="pattern/weights_transpose/MinusOne/shapes_concat" type="Concat" version="opset1">
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
		<layer id="86" name="pattern/flatten_fc_input" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>6248</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="pattern/WithoutBiases" type="MatMul" version="opset1">
			<data transpose_a="False" transpose_b="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6248</dim>
				</port>
				<port id="1">
					<dim>128</dim>
					<dim>6248</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="72/Output_0/Data_/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4814512" shape="1,128" size="512"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="pattern" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="reshape/Cast_12921_const" type="Const" version="opset1">
			<data element_type="i64" offset="4815024" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="tile/tiles1073_const" type="Const" version="opset1">
			<data element_type="i64" offset="4815056" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="tile1071" type="Tile" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
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
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="concat" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="61/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="4815088" shape="71,199,1,1" size="56516"/>
			<output>
				<port id="1" precision="FP32">
					<dim>71</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="result/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>71</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="result/Dims763/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4871604" shape="1,71,1,1" size="284"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="result" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="p_result/Cast_12919_const" type="Const" version="opset1">
			<data element_type="i64" offset="4871888" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="p_result" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="reshape2/Cast_12923_const" type="Const" version="opset1">
			<data element_type="i64" offset="4871920" shape="3" size="24"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="reshape2" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="seq_ind" type="Parameter" version="opset1">
			<data element_type="f32" shape="88,1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>88</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="decode" type="CTCGreedyDecoder" version="opset1">
			<data ctc_merge_repeated="1"/>
			<input>
				<port id="0">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
				</port>
				<port id="1">
					<dim>88</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="decode/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="8" to-port="0"/>
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
		<edge from-layer="20" from-port="2" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="23" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="1"/>
		<edge from-layer="23" from-port="2" to-layer="25" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="1"/>
		<edge from-layer="25" from-port="2" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="1"/>
		<edge from-layer="29" from-port="2" to-layer="31" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="31" to-port="1"/>
		<edge from-layer="31" from-port="2" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="1"/>
		<edge from-layer="34" from-port="2" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="1"/>
		<edge from-layer="36" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="39" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="39" to-port="1"/>
		<edge from-layer="39" from-port="2" to-layer="41" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="1"/>
		<edge from-layer="41" from-port="2" to-layer="42" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="44" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="44" to-port="1"/>
		<edge from-layer="44" from-port="2" to-layer="46" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="46" to-port="1"/>
		<edge from-layer="46" from-port="2" to-layer="47" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="49" to-port="0"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="1"/>
		<edge from-layer="49" from-port="2" to-layer="51" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="51" to-port="1"/>
		<edge from-layer="51" from-port="2" to-layer="52" to-port="0"/>
		<edge from-layer="52" from-port="1" to-layer="54" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="54" to-port="1"/>
		<edge from-layer="54" from-port="2" to-layer="56" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="56" to-port="1"/>
		<edge from-layer="56" from-port="2" to-layer="57" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="59" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="1"/>
		<edge from-layer="59" from-port="2" to-layer="61" to-port="0"/>
		<edge from-layer="60" from-port="1" to-layer="61" to-port="1"/>
		<edge from-layer="61" from-port="2" to-layer="62" to-port="0"/>
		<edge from-layer="62" from-port="1" to-layer="64" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="1"/>
		<edge from-layer="64" from-port="2" to-layer="66" to-port="0"/>
		<edge from-layer="65" from-port="1" to-layer="66" to-port="1"/>
		<edge from-layer="66" from-port="2" to-layer="67" to-port="0"/>
		<edge from-layer="67" from-port="1" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="1" to-layer="70" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="72" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="72" to-port="1"/>
		<edge from-layer="72" from-port="2" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="75" to-port="0"/>
		<edge from-layer="74" from-port="1" to-layer="75" to-port="1"/>
		<edge from-layer="75" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="76" from-port="1" to-layer="77" to-port="1"/>
		<edge from-layer="77" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="80" from-port="1" to-layer="81" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="84" to-port="0"/>
		<edge from-layer="82" from-port="1" to-layer="84" to-port="1"/>
		<edge from-layer="83" from-port="1" to-layer="84" to-port="2"/>
		<edge from-layer="79" from-port="1" to-layer="85" to-port="0"/>
		<edge from-layer="84" from-port="3" to-layer="85" to-port="1"/>
		<edge from-layer="78" from-port="1" to-layer="86" to-port="0"/>
		<edge from-layer="85" from-port="2" to-layer="86" to-port="1"/>
		<edge from-layer="86" from-port="2" to-layer="87" to-port="0"/>
		<edge from-layer="80" from-port="1" to-layer="87" to-port="1"/>
		<edge from-layer="87" from-port="2" to-layer="89" to-port="0"/>
		<edge from-layer="88" from-port="1" to-layer="89" to-port="1"/>
		<edge from-layer="89" from-port="2" to-layer="91" to-port="0"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="1"/>
		<edge from-layer="91" from-port="2" to-layer="93" to-port="0"/>
		<edge from-layer="92" from-port="1" to-layer="93" to-port="1"/>
		<edge from-layer="78" from-port="1" to-layer="94" to-port="0"/>
		<edge from-layer="93" from-port="2" to-layer="94" to-port="1"/>
		<edge from-layer="94" from-port="2" to-layer="96" to-port="0"/>
		<edge from-layer="95" from-port="1" to-layer="96" to-port="1"/>
		<edge from-layer="96" from-port="2" to-layer="98" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="98" to-port="1"/>
		<edge from-layer="98" from-port="2" to-layer="100" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="100" to-port="1"/>
		<edge from-layer="100" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="101" from-port="1" to-layer="102" to-port="1"/>
		<edge from-layer="102" from-port="2" to-layer="104" to-port="0"/>
		<edge from-layer="103" from-port="0" to-layer="104" to-port="1"/>
		<edge from-layer="104" from-port="2" to-layer="105" to-port="0"/>
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
			<framework value="caffe"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input value="data,seq_ind"/>
			<input_model value="DIR/lprnet.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/lprnet_deploy.prototxt"/>
			<input_shape value="[1,3,24,94],[88,1]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{}"/>
			<mean_values value="()"/>
			<model_name value="license-plate-recognition-barrier-0001"/>
			<output value="['decode']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'data': array([ 1,  3, 24, 94]), 'seq_ind': array([88,  1])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="()"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
