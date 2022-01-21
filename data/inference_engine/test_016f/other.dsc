<?xml version="1.0" ?>
<net name="age_gender" version="10">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,62,62"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_13521354_const" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="48,3,3,3" size="5184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="data_add_1239/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5184" shape="1,48,1,1" size="192"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv1/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
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
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="pool1" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="relu1" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Mul1_/Fused_Mul_13401342_const" type="Const" version="opset1">
			<data element_type="f32" offset="5376" shape="64,48,3,3" size="110592"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="conv2/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="data_add_12421247/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="115968" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv2/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
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
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="pool2" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="relu2" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Mul1_1119/Fused_Mul_13441346_const" type="Const" version="opset1">
			<data element_type="f32" offset="116224" shape="96,64,3,3" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="conv3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1">
					<dim>96</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="data_add_12501255/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="337408" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="conv3/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
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
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="pool3" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="relu3" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="Mul1_1131/Fused_Mul_13481350_const" type="Const" version="opset1">
			<data element_type="f32" offset="337792" shape="192,96,3,3" size="663552"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="conv4/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
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
		<layer id="21" name="data_add_12581263/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1001344" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="conv4/Fused_Add_" type="Add" version="opset1">
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
		<layer id="23" name="relu4" type="ReLU" version="opset1">
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
		<layer id="24" name="44/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="1002112" shape="256,192,3,3" size="1769472"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="conv5/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="conv5/Dims770/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2771584" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="conv5" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
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
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="relu5" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="64/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2772608" shape="256,256,3,3" size="2359296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="age_conv1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="age_conv1/Dims776/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5131904" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="age_conv1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
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
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="relu6_a" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="62/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="5132928" shape="512,256,1,1" size="524288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="age_conv2/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="age_conv2/Dims782/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5657216" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="age_conv2" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="relu7_a" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="56/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="5659264" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="age_conv3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="age_conv3/Dims764/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5661312" shape="1,1,1,1" size="4"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="age_conv3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
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
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="age_conv3/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
		<layer id="44" name="70/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="5661316" shape="256,256,3,3" size="2359296"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="gender_conv1/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="gender_conv1/Dims788/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8020612" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="gender_conv1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
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
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="relu6_g" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="38/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="8021636" shape="512,256,1,1" size="524288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="gender_conv2/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="gender_conv2/Dims758/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8545924" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="gender_conv2" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="relu7_g" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="72/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="8547972" shape="2,512,1,1" size="4096"/>
			<output>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="gender_conv3/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>2</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="gender_conv3/Dims794/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8552068" shape="1,2,1,1" size="8"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="gender_conv3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="prob" type="SoftMax" version="opset1">
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
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="prob/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
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
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="14" to-port="1"/>
		<edge from-layer="14" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="1"/>
		<edge from-layer="16" from-port="2" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="1" to-layer="20" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="20" to-port="1"/>
		<edge from-layer="20" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="1"/>
		<edge from-layer="25" from-port="2" to-layer="27" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="27" to-port="1"/>
		<edge from-layer="27" from-port="2" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="30" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="1"/>
		<edge from-layer="30" from-port="2" to-layer="32" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="35" to-port="0"/>
		<edge from-layer="34" from-port="1" to-layer="35" to-port="1"/>
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="1" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="40" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="40" to-port="1"/>
		<edge from-layer="40" from-port="2" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="45" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="45" to-port="1"/>
		<edge from-layer="45" from-port="2" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="1" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="1" to-layer="50" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="50" to-port="1"/>
		<edge from-layer="50" from-port="2" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="1" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="55" to-port="0"/>
		<edge from-layer="54" from-port="1" to-layer="55" to-port="1"/>
		<edge from-layer="55" from-port="2" to-layer="57" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="57" to-port="1"/>
		<edge from-layer="57" from-port="2" to-layer="58" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2020.1.0-55-g3b9c198827"/>
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
			<input_model value="DIR/age_gender_net.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/age_gender_net.prototxt"/>
			<input_shape value="[1,3,62,62]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'scale': array([254.99991075]), 'mean': None}}"/>
			<mean_values value="()"/>
			<model_name value="age-gender-recognition-retail-0013"/>
			<move_to_preprocess value="False"/>
			<output value="['prob', 'age_conv3']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'data': array([ 1,  3, 62, 62])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="data[254.99991075003123]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
