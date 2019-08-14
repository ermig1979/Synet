<?xml version="1.0" ?>
<net batch="1" name="LPRNet" version="4">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="6912"/>
				<biases offset="6912" size="256"/>
			</blobs>
		</layer>
		<layer id="2" name="relu_conv1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="pool1" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>24</dim>
					<dim>94</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="fire1_small/conv_reduce" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7168" size="8192"/>
				<biases offset="15360" size="128"/>
			</blobs>
		</layer>
		<layer id="5" name="fire1_small/relu_conv_reduce" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="fire1_small/conv_3x1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,1" output="32" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
			<blobs>
				<weights offset="15488" size="12288"/>
				<biases offset="27776" size="128"/>
			</blobs>
		</layer>
		<layer id="7" name="fire1_small/relu_conv_3x1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="fire1_small/conv_1x3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,3" output="32" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
			<blobs>
				<weights offset="27904" size="12288"/>
				<biases offset="40192" size="128"/>
			</blobs>
		</layer>
		<layer id="9" name="fire1_small/relu_conv_1x3" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="fire1_small/conv_expand" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
			<blobs>
				<weights offset="40320" size="16384"/>
				<biases offset="56704" size="512"/>
			</blobs>
		</layer>
		<layer id="11" name="fire1_small/relu_conv_expand" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="pool2" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>22</dim>
					<dim>92</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="fire2_small/conv_reduce" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="57216" size="32768"/>
				<biases offset="89984" size="256"/>
			</blobs>
		</layer>
		<layer id="14" name="fire2_small/relu_conv_reduce" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="fire2_small/conv_3x1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,1" output="64" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="90240" size="49152"/>
				<biases offset="139392" size="256"/>
			</blobs>
		</layer>
		<layer id="16" name="fire2_small/relu_conv_3x1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="fire2_small/conv_1x3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,3" output="64" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="139648" size="49152"/>
				<biases offset="188800" size="256"/>
			</blobs>
		</layer>
		<layer id="18" name="fire2_small/relu_conv_1x3" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="fire2_small/conv_expand" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="189056" size="65536"/>
				<biases offset="254592" size="1024"/>
			</blobs>
		</layer>
		<layer id="20" name="fire2_small/relu_conv_expand" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="fire3_small/conv_reduce" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="255616" size="65536"/>
				<biases offset="321152" size="256"/>
			</blobs>
		</layer>
		<layer id="22" name="fire3_small/relu_conv_reduce" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="fire3_small/conv_3x1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,1" output="64" pads_begin="1,0" pads_end="1,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="321408" size="49152"/>
				<biases offset="370560" size="256"/>
			</blobs>
		</layer>
		<layer id="24" name="fire3_small/relu_conv_3x1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="fire3_small/conv_1x3" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,3" output="64" pads_begin="0,1" pads_end="0,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="370816" size="49152"/>
				<biases offset="419968" size="256"/>
			</blobs>
		</layer>
		<layer id="26" name="fire3_small/relu_conv_1x3" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="fire3_small/conv_expand" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
			<blobs>
				<weights offset="420224" size="65536"/>
				<biases offset="485760" size="1024"/>
			</blobs>
		</layer>
		<layer id="28" name="fire3_small/relu_conv_expand" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="pool3" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>11</dim>
					<dim>90</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="conv2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="5,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
			<blobs>
				<weights offset="486784" size="655360"/>
				<biases offset="1142144" size="512"/>
			</blobs>
		</layer>
		<layer id="31" name="relu_conv2" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="conv3_w" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,13" output="71" pads_begin="0,6" pads_end="0,6" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1142656" size="472576"/>
				<biases offset="1615232" size="284"/>
			</blobs>
		</layer>
		<layer id="33" name="relu_conv3_w" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="pattern" precision="FP32" type="FullyConnected">
			<data out-size="128"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1615516" size="3198976"/>
				<biases offset="4814492" size="512"/>
			</blobs>
		</layer>
		<layer id="35" name="reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4815004" size="16"/>
			</blobs>
		</layer>
		<layer id="36" name="reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="tile" precision="FP32" type="Tile">
			<data axis="3" tiles="88"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="concat" precision="FP32" type="Concat">
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
				<port id="2">
					<dim>1</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="result" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="71" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4815020" size="56516"/>
				<biases offset="4871536" size="284"/>
			</blobs>
		</layer>
		<layer id="40" name="p_result" precision="FP32" type="Permute">
			<data order="3,0,1,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="reshape2/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4871820" size="12"/>
			</blobs>
		</layer>
		<layer id="42" name="reshape2" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
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
				<port id="2">
					<dim>88</dim>
					<dim>1</dim>
					<dim>71</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="seq_ind" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>88</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="decode" precision="FP32" type="CTCGreedyDecoder">
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
				<port id="2">
					<dim>1</dim>
					<dim>88</dim>
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
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="3" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="1" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="3" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="1" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="3" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="1" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="3" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="1" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="3" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="3" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="3" to-layer="31" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="3" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="3" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="1"/>
		<edge from-layer="36" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="38" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="38" to-port="1"/>
		<edge from-layer="38" from-port="2" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="3" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="44" to-port="0"/>
		<edge from-layer="43" from-port="0" to-layer="44" to-port="1"/>
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
			<input value="data,seq_ind"/>
			<input_model value="DIR/lprnet.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/lprnet_deploy.prototxt"/>
			<input_shape value="[1,3,24,94],[88,1]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{}"/>
			<mean_values value="()"/>
			<model_name value="license-plate-recognition-barrier-0001"/>
			<move_to_preprocess value="False"/>
			<output value="['decode']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'seq_ind': array([88,  1]), 'data': array([ 1,  3, 24, 94])}"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="()"/>
			<silent value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, finegrain_fusing, freeze_placeholder_with_value, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
