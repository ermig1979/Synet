<?xml version="1.0" ?>
<net name="torch-jit-export" version="11">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,3,72,72" element_type="f32"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="input"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32" names="input">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
					<rt_info>
						<attribute name="layout" version="0" layout="[N,C,H,W]"/>
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="1" name="Gather_4078" type="Const" version="opset1">
			<data element_type="f32" shape="1, 3, 1, 1" offset="0" size="12"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Subtract_302" type="Subtract" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Subtract_302"/>
				<attribute name="preprocessing" version="0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>72</dim>
					<dim>72</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Gather_4081" type="Const" version="opset1">
			<data element_type="f32" shape="64, 3, 7, 7" offset="12" size="37632"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>3</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Multiply_3599" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="3, 3" pads_end="3, 3" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="127, 128, Constant_303, Divide_304, model._resnet.bn1.bias, model._resnet.bn1.running_mean, model._resnet.bn1.running_var, model._resnet.bn1.weight, model._resnet.conv1.weight"/>
				<attribute name="preprocessing" version="0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>72</dim>
					<dim>72</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>3</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Constant_3604" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="37644" size="256"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="128" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="128, model._resnet.bn1.bias, model._resnet.bn1.running_mean, model._resnet.bn1.running_var, model._resnet.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="128">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="129" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="129"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="129">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="130" type="MaxPool" version="opset8">
			<data strides="2, 2" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" kernel="3, 3" rounding_type="floor" auto_pad="explicit" index_element_type="i64" axis="0"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="130"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>36</dim>
					<dim>36</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="130">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Multiply_3761" type="Const" version="opset1">
			<data element_type="f32" shape="64, 64, 3, 3" offset="37900" size="147456"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="Multiply_3606" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="131, 132, model._resnet.layer1.0.bn1.bias, model._resnet.layer1.0.bn1.running_mean, model._resnet.layer1.0.bn1.running_var, model._resnet.layer1.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="Constant_3611" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="185356" size="256"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="132" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="132, model._resnet.layer1.0.bn1.bias, model._resnet.layer1.0.bn1.running_mean, model._resnet.layer1.0.bn1.running_var, model._resnet.layer1.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="132">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="133" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="133"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="133">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="Multiply_3770" type="Const" version="opset1">
			<data element_type="f32" shape="64, 64, 3, 3" offset="185612" size="147456"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="Multiply_3613" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="134, 135, model._resnet.layer1.0.bn2.bias, model._resnet.layer1.0.bn2.running_mean, model._resnet.layer1.0.bn2.running_var, model._resnet.layer1.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="Constant_3618" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="333068" size="256"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="135" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="135, model._resnet.layer1.0.bn2.bias, model._resnet.layer1.0.bn2.running_mean, model._resnet.layer1.0.bn2.running_var, model._resnet.layer1.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="135">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="136" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="136"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="136">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="137" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="137"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="137">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="Multiply_3779" type="Const" version="opset1">
			<data element_type="f32" shape="64, 64, 3, 3" offset="333324" size="147456"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="Multiply_3620" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="138, 139, model._resnet.layer1.1.bn1.bias, model._resnet.layer1.1.bn1.running_mean, model._resnet.layer1.1.bn1.running_var, model._resnet.layer1.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="Constant_3625" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="480780" size="256"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="139" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="139, model._resnet.layer1.1.bn1.bias, model._resnet.layer1.1.bn1.running_mean, model._resnet.layer1.1.bn1.running_var, model._resnet.layer1.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="139">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="140" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="140"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="140">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="Multiply_3788" type="Const" version="opset1">
			<data element_type="f32" shape="64, 64, 3, 3" offset="481036" size="147456"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="Multiply_3627" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="141, 142, model._resnet.layer1.1.bn2.bias, model._resnet.layer1.1.bn2.running_mean, model._resnet.layer1.1.bn2.running_var, model._resnet.layer1.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="Constant_3632" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="628492" size="256"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="142" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="142, model._resnet.layer1.1.bn2.bias, model._resnet.layer1.1.bn2.running_mean, model._resnet.layer1.1.bn2.running_var, model._resnet.layer1.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="142">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="143" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="143"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="143">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="144" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="144"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="144">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="Multiply_3797" type="Const" version="opset1">
			<data element_type="f32" shape="128, 64, 3, 3" offset="628748" size="294912"/>
			<output>
				<port id="0" precision="FP32">
					<dim>128</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="Multiply_3634" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="145, 146, model._resnet.layer2.0.bn1.bias, model._resnet.layer2.0.bn1.running_mean, model._resnet.layer2.0.bn1.running_var, model._resnet.layer2.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="Constant_3639" type="Const" version="opset1">
			<data element_type="f32" shape="1, 128, 1, 1" offset="923660" size="512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="146" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="146, model._resnet.layer2.0.bn1.bias, model._resnet.layer2.0.bn1.running_mean, model._resnet.layer2.0.bn1.running_var, model._resnet.layer2.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="146">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="147" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="147"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="147">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="Multiply_3806" type="Const" version="opset1">
			<data element_type="f32" shape="128, 128, 3, 3" offset="924172" size="589824"/>
			<output>
				<port id="0" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="Multiply_3641" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="148, 149, model._resnet.layer2.0.bn2.bias, model._resnet.layer2.0.bn2.running_mean, model._resnet.layer2.0.bn2.running_var, model._resnet.layer2.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="Constant_3646" type="Const" version="opset1">
			<data element_type="f32" shape="1, 128, 1, 1" offset="1513996" size="512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="149" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="149, model._resnet.layer2.0.bn2.bias, model._resnet.layer2.0.bn2.running_mean, model._resnet.layer2.0.bn2.running_var, model._resnet.layer2.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="149">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="Multiply_3815" type="Const" version="opset1">
			<data element_type="f32" shape="128, 64, 1, 1" offset="1514508" size="32768"/>
			<output>
				<port id="0" precision="FP32">
					<dim>128</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="Multiply_3648" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="150, 151, model._resnet.layer2.0.downsample.1.bias, model._resnet.layer2.0.downsample.1.running_mean, model._resnet.layer2.0.downsample.1.running_var, model._resnet.layer2.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>18</dim>
					<dim>18</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="Constant_3653" type="Const" version="opset1">
			<data element_type="f32" shape="1, 128, 1, 1" offset="1547276" size="512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="151" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="151, model._resnet.layer2.0.downsample.1.bias, model._resnet.layer2.0.downsample.1.running_mean, model._resnet.layer2.0.downsample.1.running_var, model._resnet.layer2.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="151">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="152" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="152"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="152">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="153" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="153"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="153">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="Multiply_3824" type="Const" version="opset1">
			<data element_type="f32" shape="128, 128, 3, 3" offset="1547788" size="589824"/>
			<output>
				<port id="0" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="Multiply_3655" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="154, 155, model._resnet.layer2.1.bn1.bias, model._resnet.layer2.1.bn1.running_mean, model._resnet.layer2.1.bn1.running_var, model._resnet.layer2.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="Constant_3660" type="Const" version="opset1">
			<data element_type="f32" shape="1, 128, 1, 1" offset="2137612" size="512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="155" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="155, model._resnet.layer2.1.bn1.bias, model._resnet.layer2.1.bn1.running_mean, model._resnet.layer2.1.bn1.running_var, model._resnet.layer2.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="155">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="156" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="156"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="156">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="Multiply_3833" type="Const" version="opset1">
			<data element_type="f32" shape="128, 128, 3, 3" offset="2138124" size="589824"/>
			<output>
				<port id="0" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="Multiply_3662" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="157, 158, model._resnet.layer2.1.bn2.bias, model._resnet.layer2.1.bn2.running_mean, model._resnet.layer2.1.bn2.running_var, model._resnet.layer2.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>128</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="Constant_3667" type="Const" version="opset1">
			<data element_type="f32" shape="1, 128, 1, 1" offset="2727948" size="512"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="158" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="158, model._resnet.layer2.1.bn2.bias, model._resnet.layer2.1.bn2.running_mean, model._resnet.layer2.1.bn2.running_var, model._resnet.layer2.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="158">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="159" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="159"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="159">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="160" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="160"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="160">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="Multiply_3842" type="Const" version="opset1">
			<data element_type="f32" shape="256, 128, 3, 3" offset="2728460" size="1179648"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="Multiply_3669" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="161, 162, model._resnet.layer3.0.bn1.bias, model._resnet.layer3.0.bn1.running_mean, model._resnet.layer3.0.bn1.running_var, model._resnet.layer3.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>128</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="Constant_3674" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="3908108" size="1024"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="162" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="162, model._resnet.layer3.0.bn1.bias, model._resnet.layer3.0.bn1.running_mean, model._resnet.layer3.0.bn1.running_var, model._resnet.layer3.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="162">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="163" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="163"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="163">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="Multiply_3851" type="Const" version="opset1">
			<data element_type="f32" shape="256, 256, 3, 3" offset="3909132" size="2359296"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="Multiply_3676" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="164, 165, model._resnet.layer3.0.bn2.bias, model._resnet.layer3.0.bn2.running_mean, model._resnet.layer3.0.bn2.running_var, model._resnet.layer3.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="Constant_3681" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="6268428" size="1024"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="165" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="165, model._resnet.layer3.0.bn2.bias, model._resnet.layer3.0.bn2.running_mean, model._resnet.layer3.0.bn2.running_var, model._resnet.layer3.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="165">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="Multiply_3860" type="Const" version="opset1">
			<data element_type="f32" shape="256, 128, 1, 1" offset="6269452" size="131072"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="Multiply_3683" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="166, 167, model._resnet.layer3.0.downsample.1.bias, model._resnet.layer3.0.downsample.1.running_mean, model._resnet.layer3.0.downsample.1.running_var, model._resnet.layer3.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>9</dim>
					<dim>9</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="Constant_3688" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="6400524" size="1024"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="167" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="167, model._resnet.layer3.0.downsample.1.bias, model._resnet.layer3.0.downsample.1.running_mean, model._resnet.layer3.0.downsample.1.running_var, model._resnet.layer3.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="167">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="168" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="168"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="168">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="169" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="169"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="169">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="Multiply_3869" type="Const" version="opset1">
			<data element_type="f32" shape="256, 256, 3, 3" offset="6401548" size="2359296"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="Multiply_3690" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="170, 171, model._resnet.layer3.1.bn1.bias, model._resnet.layer3.1.bn1.running_mean, model._resnet.layer3.1.bn1.running_var, model._resnet.layer3.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="Constant_3695" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="8760844" size="1024"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="171" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="171, model._resnet.layer3.1.bn1.bias, model._resnet.layer3.1.bn1.running_mean, model._resnet.layer3.1.bn1.running_var, model._resnet.layer3.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="171">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="172" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="172"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="172">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="Multiply_3878" type="Const" version="opset1">
			<data element_type="f32" shape="256, 256, 3, 3" offset="8761868" size="2359296"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="Multiply_3697" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="173, 174, model._resnet.layer3.1.bn2.bias, model._resnet.layer3.1.bn2.running_mean, model._resnet.layer3.1.bn2.running_var, model._resnet.layer3.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="Constant_3702" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="11121164" size="1024"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="174" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="174, model._resnet.layer3.1.bn2.bias, model._resnet.layer3.1.bn2.running_mean, model._resnet.layer3.1.bn2.running_var, model._resnet.layer3.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="174">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="175" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="175"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="175">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="176" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="176"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="176">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="Multiply_3887" type="Const" version="opset1">
			<data element_type="f32" shape="512, 256, 3, 3" offset="11122188" size="4718592"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="Multiply_3704" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="177, 178, model._resnet.layer4.0.bn1.bias, model._resnet.layer4.0.bn1.running_mean, model._resnet.layer4.0.bn1.running_var, model._resnet.layer4.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="Constant_3709" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="15840780" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="178" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="178, model._resnet.layer4.0.bn1.bias, model._resnet.layer4.0.bn1.running_mean, model._resnet.layer4.0.bn1.running_var, model._resnet.layer4.0.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="178">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="179" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="179"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="179">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="Multiply_3896" type="Const" version="opset1">
			<data element_type="f32" shape="512, 512, 3, 3" offset="15842828" size="9437184"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="Multiply_3711" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="180, 181, model._resnet.layer4.0.bn2.bias, model._resnet.layer4.0.bn2.running_mean, model._resnet.layer4.0.bn2.running_var, model._resnet.layer4.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="Constant_3716" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="25280012" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="181" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="181, model._resnet.layer4.0.bn2.bias, model._resnet.layer4.0.bn2.running_mean, model._resnet.layer4.0.bn2.running_var, model._resnet.layer4.0.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="181">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="Multiply_3905" type="Const" version="opset1">
			<data element_type="f32" shape="512, 256, 1, 1" offset="25282060" size="524288"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="Multiply_3718" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="182, 183, model._resnet.layer4.0.downsample.1.bias, model._resnet.layer4.0.downsample.1.running_mean, model._resnet.layer4.0.downsample.1.running_var, model._resnet.layer4.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="Constant_3723" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="25806348" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="183" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="183, model._resnet.layer4.0.downsample.1.bias, model._resnet.layer4.0.downsample.1.running_mean, model._resnet.layer4.0.downsample.1.running_var, model._resnet.layer4.0.downsample.1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="183">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="184" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="184"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="184">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="185" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="185"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="185">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="Multiply_3914" type="Const" version="opset1">
			<data element_type="f32" shape="512, 512, 3, 3" offset="25808396" size="9437184"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="Multiply_3725" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="186, 187, model._resnet.layer4.1.bn1.bias, model._resnet.layer4.1.bn1.running_mean, model._resnet.layer4.1.bn1.running_var, model._resnet.layer4.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="Constant_3730" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="35245580" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="187" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="187, model._resnet.layer4.1.bn1.bias, model._resnet.layer4.1.bn1.running_mean, model._resnet.layer4.1.bn1.running_var, model._resnet.layer4.1.bn1.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="187">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="188" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="188"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="188">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="Multiply_3923" type="Const" version="opset1">
			<data element_type="f32" shape="512, 512, 3, 3" offset="35247628" size="9437184"/>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="Multiply_3732" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="189, 190, model._resnet.layer4.1.bn2.bias, model._resnet.layer4.1.bn2.running_mean, model._resnet.layer4.1.bn2.running_var, model._resnet.layer4.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="Constant_3737" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="44684812" size="2048"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="190" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="190, model._resnet.layer4.1.bn2.bias, model._resnet.layer4.1.bn2.running_mean, model._resnet.layer4.1.bn2.running_var, model._resnet.layer4.1.bn2.weight"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="190">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="191" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="191"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="191">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="192" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="192"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="192">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="Range_220" type="Const" version="opset1">
			<data element_type="i64" shape="2" offset="44686860" size="16"/>
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="193" type="ReduceMean" version="opset1">
			<data keep_dims="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="193, Range_220"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="193">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="Constant_245" type="Const" version="opset1">
			<data element_type="i64" shape="2" offset="44686876" size="16"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Constant_245"/>
			</rt_info>
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="194" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="194"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="194">
					<dim>1</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="model.fc_veh_type.weight" type="Const" version="opset1">
			<data element_type="f32" shape="4, 512" offset="44686892" size="8192"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="model.fc_veh_type.weight"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32" names="model.fc_veh_type.weight">
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="MatMul_259" type="MatMul" version="opset1">
			<data transpose_a="false" transpose_b="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="MatMul_259"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="Constant_4082" type="Const" version="opset1">
			<data element_type="f32" shape="1, 4" offset="44695084" size="16"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="196" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="196, Multiply_260"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="196">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="Constant_277" type="Const" version="opset1">
			<data element_type="i64" shape="2" offset="44686876" size="16"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Constant_277"/>
			</rt_info>
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="Reshape_278" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Reshape_278"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="Softmax_284" type="SoftMax" version="opset8">
			<data axis="1"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Softmax_284"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="ShapeOf_285" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="ShapeOf_285"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="type" type="Reshape" version="opset1">
			<data special_zero="false"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="type"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="type">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="model.fc_veh_color.weight" type="Const" version="opset1">
			<data element_type="f32" shape="7, 512" offset="44695100" size="14336"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="model.fc_veh_color.weight"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32" names="model.fc_veh_color.weight">
					<dim>7</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="MatMul_254" type="MatMul" version="opset1">
			<data transpose_a="false" transpose_b="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="MatMul_254"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>7</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="Constant_4083" type="Const" version="opset1">
			<data element_type="f32" shape="1, 7" offset="44709436" size="28"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="195" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="195, Multiply_255"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="195">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="Constant_262" type="Const" version="opset1">
			<data element_type="i64" shape="2" offset="44686876" size="16"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Constant_262"/>
			</rt_info>
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="Reshape_263" type="Reshape" version="opset1">
			<data special_zero="true"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Reshape_263"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="Softmax_269" type="SoftMax" version="opset8">
			<data axis="1"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="Softmax_269"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="ShapeOf_270" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="ShapeOf_270"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="color" type="Reshape" version="opset1">
			<data special_zero="false"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="color"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="color">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="color/sink_port_0" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="color/sink_port_0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
		<layer id="122" name="type/sink_port_0" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="type/sink_port_0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="6" to-port="1"/>
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="18" to-port="1"/>
		<edge from-layer="9" from-port="0" to-layer="10" to-port="1"/>
		<edge from-layer="10" from-port="2" to-layer="12" to-port="0"/>
		<edge from-layer="11" from-port="0" to-layer="12" to-port="1"/>
		<edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="15" to-port="0"/>
		<edge from-layer="14" from-port="0" to-layer="15" to-port="1"/>
		<edge from-layer="15" from-port="2" to-layer="17" to-port="0"/>
		<edge from-layer="16" from-port="0" to-layer="17" to-port="1"/>
		<edge from-layer="17" from-port="2" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="21" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="29" to-port="1"/>
		<edge from-layer="20" from-port="0" to-layer="21" to-port="1"/>
		<edge from-layer="21" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="22" from-port="0" to-layer="23" to-port="1"/>
		<edge from-layer="23" from-port="2" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="26" to-port="0"/>
		<edge from-layer="25" from-port="0" to-layer="26" to-port="1"/>
		<edge from-layer="26" from-port="2" to-layer="28" to-port="0"/>
		<edge from-layer="27" from-port="0" to-layer="28" to-port="1"/>
		<edge from-layer="28" from-port="2" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="2" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="41" to-port="0"/>
		<edge from-layer="31" from-port="0" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="34" to-port="0"/>
		<edge from-layer="33" from-port="0" to-layer="34" to-port="1"/>
		<edge from-layer="34" from-port="2" to-layer="35" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="0" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="39" to-port="0"/>
		<edge from-layer="38" from-port="0" to-layer="39" to-port="1"/>
		<edge from-layer="39" from-port="2" to-layer="44" to-port="0"/>
		<edge from-layer="40" from-port="0" to-layer="41" to-port="1"/>
		<edge from-layer="41" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="42" from-port="0" to-layer="43" to-port="1"/>
		<edge from-layer="43" from-port="2" to-layer="44" to-port="1"/>
		<edge from-layer="44" from-port="2" to-layer="45" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="55" to-port="1"/>
		<edge from-layer="46" from-port="0" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="48" from-port="0" to-layer="49" to-port="1"/>
		<edge from-layer="49" from-port="2" to-layer="50" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="0" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="54" to-port="0"/>
		<edge from-layer="53" from-port="0" to-layer="54" to-port="1"/>
		<edge from-layer="54" from-port="2" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="2" to-layer="56" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="58" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="67" to-port="0"/>
		<edge from-layer="57" from-port="0" to-layer="58" to-port="1"/>
		<edge from-layer="58" from-port="2" to-layer="60" to-port="0"/>
		<edge from-layer="59" from-port="0" to-layer="60" to-port="1"/>
		<edge from-layer="60" from-port="2" to-layer="61" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="63" to-port="0"/>
		<edge from-layer="62" from-port="0" to-layer="63" to-port="1"/>
		<edge from-layer="63" from-port="2" to-layer="65" to-port="0"/>
		<edge from-layer="64" from-port="0" to-layer="65" to-port="1"/>
		<edge from-layer="65" from-port="2" to-layer="70" to-port="0"/>
		<edge from-layer="66" from-port="0" to-layer="67" to-port="1"/>
		<edge from-layer="67" from-port="2" to-layer="69" to-port="0"/>
		<edge from-layer="68" from-port="0" to-layer="69" to-port="1"/>
		<edge from-layer="69" from-port="2" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="71" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="73" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="81" to-port="1"/>
		<edge from-layer="72" from-port="0" to-layer="73" to-port="1"/>
		<edge from-layer="73" from-port="2" to-layer="75" to-port="0"/>
		<edge from-layer="74" from-port="0" to-layer="75" to-port="1"/>
		<edge from-layer="75" from-port="2" to-layer="76" to-port="0"/>
		<edge from-layer="76" from-port="1" to-layer="78" to-port="0"/>
		<edge from-layer="77" from-port="0" to-layer="78" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="80" to-port="0"/>
		<edge from-layer="79" from-port="0" to-layer="80" to-port="1"/>
		<edge from-layer="80" from-port="2" to-layer="81" to-port="0"/>
		<edge from-layer="81" from-port="2" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="1" to-layer="84" to-port="0"/>
		<edge from-layer="82" from-port="1" to-layer="93" to-port="0"/>
		<edge from-layer="83" from-port="0" to-layer="84" to-port="1"/>
		<edge from-layer="84" from-port="2" to-layer="86" to-port="0"/>
		<edge from-layer="85" from-port="0" to-layer="86" to-port="1"/>
		<edge from-layer="86" from-port="2" to-layer="87" to-port="0"/>
		<edge from-layer="87" from-port="1" to-layer="89" to-port="0"/>
		<edge from-layer="88" from-port="0" to-layer="89" to-port="1"/>
		<edge from-layer="89" from-port="2" to-layer="91" to-port="0"/>
		<edge from-layer="90" from-port="0" to-layer="91" to-port="1"/>
		<edge from-layer="91" from-port="2" to-layer="96" to-port="0"/>
		<edge from-layer="92" from-port="0" to-layer="93" to-port="1"/>
		<edge from-layer="93" from-port="2" to-layer="95" to-port="0"/>
		<edge from-layer="94" from-port="0" to-layer="95" to-port="1"/>
		<edge from-layer="95" from-port="2" to-layer="96" to-port="1"/>
		<edge from-layer="96" from-port="2" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="99" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="107" to-port="1"/>
		<edge from-layer="98" from-port="0" to-layer="99" to-port="1"/>
		<edge from-layer="99" from-port="2" to-layer="101" to-port="0"/>
		<edge from-layer="100" from-port="0" to-layer="101" to-port="1"/>
		<edge from-layer="101" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="1" to-layer="104" to-port="0"/>
		<edge from-layer="103" from-port="0" to-layer="104" to-port="1"/>
		<edge from-layer="104" from-port="2" to-layer="106" to-port="0"/>
		<edge from-layer="105" from-port="0" to-layer="106" to-port="1"/>
		<edge from-layer="106" from-port="2" to-layer="107" to-port="0"/>
		<edge from-layer="107" from-port="2" to-layer="108" to-port="0"/>
		<edge from-layer="108" from-port="1" to-layer="110" to-port="0"/>
		<edge from-layer="109" from-port="0" to-layer="110" to-port="1"/>
		<edge from-layer="110" from-port="2" to-layer="112" to-port="0"/>
		<edge from-layer="111" from-port="0" to-layer="112" to-port="1"/>
		<edge from-layer="112" from-port="2" to-layer="114" to-port="0"/>
		<edge from-layer="112" from-port="2" to-layer="124" to-port="0"/>
		<edge from-layer="113" from-port="0" to-layer="114" to-port="1"/>
		<edge from-layer="114" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="115" from-port="0" to-layer="116" to-port="1"/>
		<edge from-layer="116" from-port="2" to-layer="118" to-port="0"/>
		<edge from-layer="116" from-port="2" to-layer="120" to-port="0"/>
		<edge from-layer="117" from-port="0" to-layer="118" to-port="1"/>
		<edge from-layer="118" from-port="2" to-layer="119" to-port="0"/>
		<edge from-layer="119" from-port="1" to-layer="121" to-port="0"/>
		<edge from-layer="120" from-port="1" to-layer="121" to-port="1"/>
		<edge from-layer="121" from-port="2" to-layer="122" to-port="0"/>
		<edge from-layer="123" from-port="0" to-layer="124" to-port="1"/>
		<edge from-layer="124" from-port="2" to-layer="126" to-port="0"/>
		<edge from-layer="125" from-port="0" to-layer="126" to-port="1"/>
		<edge from-layer="126" from-port="2" to-layer="128" to-port="0"/>
		<edge from-layer="126" from-port="2" to-layer="130" to-port="0"/>
		<edge from-layer="127" from-port="0" to-layer="128" to-port="1"/>
		<edge from-layer="128" from-port="2" to-layer="129" to-port="0"/>
		<edge from-layer="129" from-port="1" to-layer="131" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="131" to-port="1"/>
		<edge from-layer="131" from-port="2" to-layer="132" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2022.1.0-6910-173f328c53d"/>
		<Runtime_version value="2022.1.0-6910-173f328c53d"/>
		<legacy_path value="False"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<compress_fp16 value="False"/>
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
			<input value="input"/>
			<input_model value="DIR/resnet18_vehicle_attributes_v04.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,72,72]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<layout value="input(nchw)"/>
			<layout_values value="{'input': {'source_layout': 'nchw', 'target_layout': None, 'is_input': True}}"/>
			<legacy_ir_generation value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'input': {'mean': array([123.675, 116.28 , 103.53 ]), 'scale': array([58.395, 57.12 , 57.375])}}"/>
			<mean_values value="input[123.675,116.28,103.53]"/>
			<model_name value="vehicle-attributes-recognition-barrier-0042"/>
			<output value="['color', 'type']"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'input': (1, 3, 72, 72)}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input[58.395,57.12,57.375]"/>
			<silent value="False"/>
			<source_layout value="()"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<target_layout value="()"/>
			<transform value=""/>
			<use_legacy_frontend value="False"/>
			<use_new_frontend value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
