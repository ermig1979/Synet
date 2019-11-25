<?xml version="1.0" ?>
<net batch="1" name="text-detection-0004" precision="FP32" version="7">
	<layers>
		<layer id="0" name="Placeholder" type="Input">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>768</dim>
					<dim>1280</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_/Fused_Mul_/FusedScaleShift_" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>768</dim>
					<dim>1280</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>768</dim>
					<dim>1280</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" precision="FP32" size="12"/>
				<biases offset="12" precision="FP32" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="model/Conv1/Conv2D" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>768</dim>
					<dim>1280</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" precision="FP32" size="5184"/>
				<biases offset="5208" precision="FP32" size="192"/>
			</blobs>
		</layer>
		<layer id="3" name="model/Conv1_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="model/expanded_conv_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="48" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5400" precision="FP32" size="1728"/>
				<biases offset="7128" precision="FP32" size="192"/>
			</blobs>
		</layer>
		<layer id="5" name="model/expanded_conv_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="model/expanded_conv_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7320" precision="FP32" size="4608"/>
				<biases offset="11928" precision="FP32" size="96"/>
			</blobs>
		</layer>
		<layer id="7" name="model/block_1_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="144" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12024" precision="FP32" size="13824"/>
				<biases offset="25848" precision="FP32" size="576"/>
			</blobs>
		</layer>
		<layer id="8" name="model/block_1_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="model/block_1_depthwise/depthwise" type="Convolution">
			<data dilations="1,1" group="144" kernel="3,3" output="144" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>384</dim>
					<dim>640</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="26424" precision="FP32" size="5184"/>
				<biases offset="31608" precision="FP32" size="576"/>
			</blobs>
		</layer>
		<layer id="10" name="model/block_1_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="model/block_1_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="32184" precision="FP32" size="18432"/>
				<biases offset="50616" precision="FP32" size="128"/>
			</blobs>
		</layer>
		<layer id="12" name="model/block_2_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="192" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="50744" precision="FP32" size="24576"/>
				<biases offset="75320" precision="FP32" size="768"/>
			</blobs>
		</layer>
		<layer id="13" name="model/block_2_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="model/block_2_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="192" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="76088" precision="FP32" size="6912"/>
				<biases offset="83000" precision="FP32" size="768"/>
			</blobs>
		</layer>
		<layer id="15" name="model/block_2_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="model/block_2_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="83768" precision="FP32" size="24576"/>
				<biases offset="108344" precision="FP32" size="128"/>
			</blobs>
		</layer>
		<layer id="17" name="model/block_2_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="model/conv2d_7/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="108472" precision="FP32" size="2048"/>
				<biases offset="110520" precision="FP32" size="64"/>
			</blobs>
		</layer>
		<layer id="19" name="model/block_3_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="192" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="110584" precision="FP32" size="24576"/>
				<biases offset="135160" precision="FP32" size="768"/>
			</blobs>
		</layer>
		<layer id="20" name="model/block_3_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="model/block_3_depthwise/depthwise" type="Convolution">
			<data dilations="1,1" group="192" kernel="3,3" output="192" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="135928" precision="FP32" size="6912"/>
				<biases offset="142840" precision="FP32" size="768"/>
			</blobs>
		</layer>
		<layer id="22" name="model/block_3_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="model/block_3_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="143608" precision="FP32" size="36864"/>
				<biases offset="180472" precision="FP32" size="192"/>
			</blobs>
		</layer>
		<layer id="24" name="model/block_4_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="288" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="180664" precision="FP32" size="55296"/>
				<biases offset="235960" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="25" name="model/block_4_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="model/block_4_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="288" kernel="3,3" output="288" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="237112" precision="FP32" size="10368"/>
				<biases offset="247480" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="27" name="model/block_4_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="model/block_4_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="248632" precision="FP32" size="55296"/>
				<biases offset="303928" precision="FP32" size="192"/>
			</blobs>
		</layer>
		<layer id="29" name="model/block_4_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="model/block_5_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="288" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="304120" precision="FP32" size="55296"/>
				<biases offset="359416" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="31" name="model/block_5_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="model/block_5_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="288" kernel="3,3" output="288" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="360568" precision="FP32" size="10368"/>
				<biases offset="370936" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="33" name="model/block_5_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="model/block_5_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="372088" precision="FP32" size="55296"/>
				<biases offset="427384" precision="FP32" size="192"/>
			</blobs>
		</layer>
		<layer id="35" name="model/block_5_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="model/conv2d_6/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="427576" precision="FP32" size="3072"/>
				<biases offset="430648" precision="FP32" size="64"/>
			</blobs>
		</layer>
		<layer id="37" name="model/block_6_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="288" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="430712" precision="FP32" size="55296"/>
				<biases offset="486008" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="38" name="model/block_6_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="model/block_6_depthwise/depthwise" type="Convolution">
			<data dilations="1,1" group="288" kernel="3,3" output="288" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="487160" precision="FP32" size="10368"/>
				<biases offset="497528" precision="FP32" size="1152"/>
			</blobs>
		</layer>
		<layer id="40" name="model/block_6_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>288</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="model/block_6_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="88" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>288</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="498680" precision="FP32" size="101376"/>
				<biases offset="600056" precision="FP32" size="352"/>
			</blobs>
		</layer>
		<layer id="42" name="model/block_7_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="528" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="600408" precision="FP32" size="185856"/>
				<biases offset="786264" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="43" name="model/block_7_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="model/block_7_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="528" kernel="3,3" output="528" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="788376" precision="FP32" size="19008"/>
				<biases offset="807384" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="45" name="model/block_7_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="model/block_7_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="88" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="809496" precision="FP32" size="185856"/>
				<biases offset="995352" precision="FP32" size="352"/>
			</blobs>
		</layer>
		<layer id="47" name="model/block_7_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="model/block_8_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="528" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="995704" precision="FP32" size="185856"/>
				<biases offset="1181560" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="49" name="model/block_8_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="model/block_8_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="528" kernel="3,3" output="528" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1183672" precision="FP32" size="19008"/>
				<biases offset="1202680" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="51" name="model/block_8_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="model/block_8_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="88" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1204792" precision="FP32" size="185856"/>
				<biases offset="1390648" precision="FP32" size="352"/>
			</blobs>
		</layer>
		<layer id="53" name="model/block_8_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="model/block_9_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="528" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1391000" precision="FP32" size="185856"/>
				<biases offset="1576856" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="55" name="model/block_9_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="model/block_9_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="528" kernel="3,3" output="528" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1578968" precision="FP32" size="19008"/>
				<biases offset="1597976" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="57" name="model/block_9_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="model/block_9_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="88" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1600088" precision="FP32" size="185856"/>
				<biases offset="1785944" precision="FP32" size="352"/>
			</blobs>
		</layer>
		<layer id="59" name="model/block_9_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="model/block_10_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="528" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>88</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1786296" precision="FP32" size="185856"/>
				<biases offset="1972152" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="61" name="model/block_10_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="model/block_10_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="528" kernel="3,3" output="528" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1974264" precision="FP32" size="19008"/>
				<biases offset="1993272" precision="FP32" size="2112"/>
			</blobs>
		</layer>
		<layer id="63" name="model/block_10_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="model/block_10_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="136" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>528</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1995384" precision="FP32" size="287232"/>
				<biases offset="2282616" precision="FP32" size="544"/>
			</blobs>
		</layer>
		<layer id="65" name="model/block_11_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="816" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2283160" precision="FP32" size="443904"/>
				<biases offset="2727064" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="66" name="model/block_11_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="model/block_11_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="816" kernel="3,3" output="816" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2730328" precision="FP32" size="29376"/>
				<biases offset="2759704" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="68" name="model/block_11_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="model/block_11_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="136" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2762968" precision="FP32" size="443904"/>
				<biases offset="3206872" precision="FP32" size="544"/>
			</blobs>
		</layer>
		<layer id="70" name="model/block_11_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="model/block_12_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="816" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3207416" precision="FP32" size="443904"/>
				<biases offset="3651320" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="72" name="model/block_12_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="model/block_12_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="816" kernel="3,3" output="816" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3654584" precision="FP32" size="29376"/>
				<biases offset="3683960" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="74" name="model/block_12_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="model/block_12_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="136" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3687224" precision="FP32" size="443904"/>
				<biases offset="4131128" precision="FP32" size="544"/>
			</blobs>
		</layer>
		<layer id="76" name="model/block_12_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="model/conv2d_5/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4131672" precision="FP32" size="8704"/>
				<biases offset="4140376" precision="FP32" size="64"/>
			</blobs>
		</layer>
		<layer id="78" name="model/block_13_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="816" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4140440" precision="FP32" size="443904"/>
				<biases offset="4584344" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="79" name="model/block_13_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="model/block_13_depthwise/depthwise" type="Convolution">
			<data dilations="1,1" group="816" kernel="3,3" output="816" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4587608" precision="FP32" size="29376"/>
				<biases offset="4616984" precision="FP32" size="3264"/>
			</blobs>
		</layer>
		<layer id="81" name="model/block_13_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>816</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="model/block_13_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="224" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>816</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4620248" precision="FP32" size="731136"/>
				<biases offset="5351384" precision="FP32" size="896"/>
			</blobs>
		</layer>
		<layer id="83" name="model/block_14_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="1344" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5352280" precision="FP32" size="1204224"/>
				<biases offset="6556504" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="84" name="model/block_14_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="model/block_14_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1344" kernel="3,3" output="1344" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6561880" precision="FP32" size="48384"/>
				<biases offset="6610264" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="86" name="model/block_14_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="model/block_14_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="224" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6615640" precision="FP32" size="1204224"/>
				<biases offset="7819864" precision="FP32" size="896"/>
			</blobs>
		</layer>
		<layer id="88" name="model/block_14_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="model/block_15_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="1344" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7820760" precision="FP32" size="1204224"/>
				<biases offset="9024984" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="90" name="model/block_15_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="model/block_15_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1344" kernel="3,3" output="1344" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9030360" precision="FP32" size="48384"/>
				<biases offset="9078744" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="92" name="model/block_15_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="model/block_15_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="224" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9084120" precision="FP32" size="1204224"/>
				<biases offset="10288344" precision="FP32" size="896"/>
			</blobs>
		</layer>
		<layer id="94" name="model/block_15_add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="model/block_16_expand/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="1344" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>224</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="10289240" precision="FP32" size="1204224"/>
				<biases offset="11493464" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="96" name="model/block_16_expand_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="model/block_16_depthwise/depthwise" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1344" kernel="3,3" output="1344" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11498840" precision="FP32" size="48384"/>
				<biases offset="11547224" precision="FP32" size="5376"/>
			</blobs>
		</layer>
		<layer id="98" name="model/block_16_depthwise_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="model/block_16_project/Conv2D" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="448" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>448</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11552600" precision="FP32" size="2408448"/>
				<biases offset="13961048" precision="FP32" size="1792"/>
			</blobs>
		</layer>
		<layer id="100" name="model/Conv_1/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="1792" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>448</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1792</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="13962840" precision="FP32" size="3211264"/>
				<biases offset="17174104" precision="FP32" size="7168"/>
			</blobs>
		</layer>
		<layer id="101" name="model/out_relu/Relu6" type="Clamp">
			<data max="6" min="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1792</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1792</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="model/conv2d_4/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1792</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="17181272" precision="FP32" size="114688"/>
				<biases offset="17295960" precision="FP32" size="64"/>
			</blobs>
		</layer>
		<layer id="103" name="model/up_sampling2d_3/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="model/add_2/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="model/up_sampling2d_4/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="model/add_3/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="model/up_sampling2d_5/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="model/link_logits_/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="model/conv2d_3/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="2" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
			<blobs>
				<weights offset="17296024" precision="FP32" size="256"/>
				<biases offset="17296280" precision="FP32" size="8"/>
			</blobs>
		</layer>
		<layer id="110" name="model/conv2d_2/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="2" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
			<blobs>
				<weights offset="17296288" precision="FP32" size="384"/>
				<biases offset="17296672" precision="FP32" size="8"/>
			</blobs>
		</layer>
		<layer id="111" name="model/conv2d_1/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="2" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>136</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
			<blobs>
				<weights offset="17296680" precision="FP32" size="1088"/>
				<biases offset="17297768" precision="FP32" size="8"/>
			</blobs>
		</layer>
		<layer id="112" name="model/conv2d/Conv2D" type="Convolution">
			<data auto_pad="valid" dilations="1,1" group="1" kernel="1,1" output="2" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1792</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</output>
			<blobs>
				<weights offset="17297776" precision="FP32" size="14336"/>
				<biases offset="17312112" precision="FP32" size="8"/>
			</blobs>
		</layer>
		<layer id="113" name="model/up_sampling2d/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>24</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="model/add/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="model/up_sampling2d_1/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48</dim>
					<dim>80</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="model/add_1/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="model/up_sampling2d_2/ResizeBilinear" type="Interp">
			<data align_corners="0" factor="2.0" height="0" pad_beg="0" pad_end="0" width="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
					<dim>160</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="model/segm_logits/add" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>192</dim>
					<dim>320</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="3" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="3" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="3" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="1"/>
		<edge from-layer="17" from-port="2" to-layer="18" to-port="0"/>
		<edge from-layer="17" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="3" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="1" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="3" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="3" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="1" to-layer="28" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="29" to-port="0"/>
		<edge from-layer="28" from-port="3" to-layer="29" to-port="1"/>
		<edge from-layer="29" from-port="2" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="3" to-layer="31" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="3" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="29" from-port="2" to-layer="35" to-port="0"/>
		<edge from-layer="34" from-port="3" to-layer="35" to-port="1"/>
		<edge from-layer="35" from-port="2" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="37" from-port="3" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="3" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="0"/>
		<edge from-layer="41" from-port="3" to-layer="42" to-port="0"/>
		<edge from-layer="42" from-port="3" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="44" to-port="0"/>
		<edge from-layer="44" from-port="3" to-layer="45" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="46" to-port="0"/>
		<edge from-layer="41" from-port="3" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="3" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="3" to-layer="49" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="50" to-port="0"/>
		<edge from-layer="50" from-port="3" to-layer="51" to-port="0"/>
		<edge from-layer="51" from-port="1" to-layer="52" to-port="0"/>
		<edge from-layer="47" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="52" from-port="3" to-layer="53" to-port="1"/>
		<edge from-layer="53" from-port="2" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="3" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="56" to-port="0"/>
		<edge from-layer="56" from-port="3" to-layer="57" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="58" to-port="0"/>
		<edge from-layer="53" from-port="2" to-layer="59" to-port="0"/>
		<edge from-layer="58" from-port="3" to-layer="59" to-port="1"/>
		<edge from-layer="59" from-port="2" to-layer="60" to-port="0"/>
		<edge from-layer="60" from-port="3" to-layer="61" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="62" to-port="0"/>
		<edge from-layer="62" from-port="3" to-layer="63" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="0"/>
		<edge from-layer="64" from-port="3" to-layer="65" to-port="0"/>
		<edge from-layer="65" from-port="3" to-layer="66" to-port="0"/>
		<edge from-layer="66" from-port="1" to-layer="67" to-port="0"/>
		<edge from-layer="67" from-port="3" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="1" to-layer="69" to-port="0"/>
		<edge from-layer="64" from-port="3" to-layer="70" to-port="0"/>
		<edge from-layer="69" from-port="3" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="71" to-port="0"/>
		<edge from-layer="71" from-port="3" to-layer="72" to-port="0"/>
		<edge from-layer="72" from-port="1" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="3" to-layer="74" to-port="0"/>
		<edge from-layer="74" from-port="1" to-layer="75" to-port="0"/>
		<edge from-layer="70" from-port="2" to-layer="76" to-port="0"/>
		<edge from-layer="75" from-port="3" to-layer="76" to-port="1"/>
		<edge from-layer="76" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="76" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="78" from-port="3" to-layer="79" to-port="0"/>
		<edge from-layer="79" from-port="1" to-layer="80" to-port="0"/>
		<edge from-layer="80" from-port="3" to-layer="81" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="3" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="3" to-layer="84" to-port="0"/>
		<edge from-layer="84" from-port="1" to-layer="85" to-port="0"/>
		<edge from-layer="85" from-port="3" to-layer="86" to-port="0"/>
		<edge from-layer="86" from-port="1" to-layer="87" to-port="0"/>
		<edge from-layer="82" from-port="3" to-layer="88" to-port="0"/>
		<edge from-layer="87" from-port="3" to-layer="88" to-port="1"/>
		<edge from-layer="88" from-port="2" to-layer="89" to-port="0"/>
		<edge from-layer="89" from-port="3" to-layer="90" to-port="0"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="0"/>
		<edge from-layer="91" from-port="3" to-layer="92" to-port="0"/>
		<edge from-layer="92" from-port="1" to-layer="93" to-port="0"/>
		<edge from-layer="88" from-port="2" to-layer="94" to-port="0"/>
		<edge from-layer="93" from-port="3" to-layer="94" to-port="1"/>
		<edge from-layer="94" from-port="2" to-layer="95" to-port="0"/>
		<edge from-layer="95" from-port="3" to-layer="96" to-port="0"/>
		<edge from-layer="96" from-port="1" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="3" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="99" to-port="0"/>
		<edge from-layer="99" from-port="3" to-layer="100" to-port="0"/>
		<edge from-layer="100" from-port="3" to-layer="101" to-port="0"/>
		<edge from-layer="101" from-port="1" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="3" to-layer="103" to-port="0"/>
		<edge from-layer="77" from-port="3" to-layer="104" to-port="0"/>
		<edge from-layer="103" from-port="1" to-layer="104" to-port="1"/>
		<edge from-layer="104" from-port="2" to-layer="105" to-port="0"/>
		<edge from-layer="36" from-port="3" to-layer="106" to-port="0"/>
		<edge from-layer="105" from-port="1" to-layer="106" to-port="1"/>
		<edge from-layer="106" from-port="2" to-layer="107" to-port="0"/>
		<edge from-layer="18" from-port="3" to-layer="108" to-port="0"/>
		<edge from-layer="107" from-port="1" to-layer="108" to-port="1"/>
		<edge from-layer="17" from-port="2" to-layer="109" to-port="0"/>
		<edge from-layer="35" from-port="2" to-layer="110" to-port="0"/>
		<edge from-layer="76" from-port="2" to-layer="111" to-port="0"/>
		<edge from-layer="101" from-port="1" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="3" to-layer="113" to-port="0"/>
		<edge from-layer="111" from-port="3" to-layer="114" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="114" to-port="1"/>
		<edge from-layer="114" from-port="2" to-layer="115" to-port="0"/>
		<edge from-layer="110" from-port="3" to-layer="116" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="116" to-port="1"/>
		<edge from-layer="116" from-port="2" to-layer="117" to-port="0"/>
		<edge from-layer="109" from-port="3" to-layer="118" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="118" to-port="1"/>
	</edges>
	<meta_data>
		<MO_version value="0.0.0-1985-g9e6b9ab51"/>
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
			<framework value="tf"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V2 value="False"/>
			<generate_deprecated_IR_V7 value="True"/>
			<generate_experimental_IR_V10 value="False"/>
			<generate_new_TI value="False"/>
			<input value="Placeholder"/>
			<input_model value="DIR/pixel_link_mobilenet_v2.pb"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,768,1280,3]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'Placeholder': {'mean': array([127.5, 127.5, 127.5]), 'scale': array([127.50000008])}}"/>
			<mean_values value="Placeholder[127.5,127.5,127.5]"/>
			<model_name value="text-detection-0004"/>
			<move_to_preprocess value="False"/>
			<output value="['model/segm_logits/add', 'model/link_logits_/add']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'Placeholder': array([   1,  768, 1280,    3])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="Placeholder[127.5000000796875]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
