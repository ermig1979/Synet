<?xml version="1.0" ?>
<net batch="1" name="face-reidentification-retail-0095" version="4">
	<layers>
		<layer id="0" name="0" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="410" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="6912"/>
				<biases offset="6936" size="256"/>
			</blobs>
		</layer>
		<layer id="3" name="412" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7192" size="4"/>
			</blobs>
		</layer>
		<layer id="4" name="413" precision="FP32" type="Convolution">
			<data dilations="1,1" group="64" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7196" size="2304"/>
				<biases offset="9500" size="256"/>
			</blobs>
		</layer>
		<layer id="5" name="415" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9756" size="4"/>
			</blobs>
		</layer>
		<layer id="6" name="416" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9760" size="32768"/>
				<biases offset="42528" size="512"/>
			</blobs>
		</layer>
		<layer id="7" name="418" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<weights offset="43040" size="4"/>
			</blobs>
		</layer>
		<layer id="8" name="419" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="43044" size="4608"/>
				<biases offset="47652" size="512"/>
			</blobs>
		</layer>
		<layer id="9" name="421" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="48164" size="4"/>
			</blobs>
		</layer>
		<layer id="10" name="422" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="48168" size="32768"/>
				<biases offset="80936" size="256"/>
			</blobs>
		</layer>
		<layer id="11" name="425" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="32,32" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="32,32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="426" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="8" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="81192" size="2048"/>
				<biases offset="83240" size="32"/>
			</blobs>
		</layer>
		<layer id="13" name="427" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="83272" size="4"/>
			</blobs>
		</layer>
		<layer id="14" name="428" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
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
				<weights offset="83276" size="2048"/>
				<biases offset="85324" size="256"/>
			</blobs>
		</layer>
		<layer id="15" name="429" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="430/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="430/Broadcast/9798" precision="FP32" type="Tile">
			<data axis="3" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="430" precision="FP32" type="Eltwise">
			<data operation="mul"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="431" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="85580" size="32768"/>
				<biases offset="118348" size="512"/>
			</blobs>
		</layer>
		<layer id="20" name="433" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="118860" size="4"/>
			</blobs>
		</layer>
		<layer id="21" name="434" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="118864" size="4608"/>
				<biases offset="123472" size="512"/>
			</blobs>
		</layer>
		<layer id="22" name="436" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="123984" size="4"/>
			</blobs>
		</layer>
		<layer id="23" name="437" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="123988" size="32768"/>
				<biases offset="156756" size="256"/>
			</blobs>
		</layer>
		<layer id="24" name="440" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="32,32" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="32,32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="441" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="8" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="157012" size="2048"/>
				<biases offset="159060" size="32"/>
			</blobs>
		</layer>
		<layer id="26" name="442" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="159092" size="4"/>
			</blobs>
		</layer>
		<layer id="27" name="443" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
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
				<weights offset="159096" size="2048"/>
				<biases offset="161144" size="256"/>
			</blobs>
		</layer>
		<layer id="28" name="444" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="445/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="445/Broadcast/9806" precision="FP32" type="Tile">
			<data axis="3" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="445" precision="FP32" type="Eltwise">
			<data operation="mul"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="446" precision="FP32" type="Eltwise">
			<data operation="sum"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="447" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="161400" size="32768"/>
				<biases offset="194168" size="512"/>
			</blobs>
		</layer>
		<layer id="34" name="449" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="194680" size="4"/>
			</blobs>
		</layer>
		<layer id="35" name="450" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="194684" size="4608"/>
				<biases offset="199292" size="512"/>
			</blobs>
		</layer>
		<layer id="36" name="452" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="199804" size="4"/>
			</blobs>
		</layer>
		<layer id="37" name="453" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="199808" size="32768"/>
				<biases offset="232576" size="256"/>
			</blobs>
		</layer>
		<layer id="38" name="456" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="32,32" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="32,32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="457" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="8" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="232832" size="2048"/>
				<biases offset="234880" size="32"/>
			</blobs>
		</layer>
		<layer id="40" name="458" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="234912" size="4"/>
			</blobs>
		</layer>
		<layer id="41" name="459" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
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
				<weights offset="234916" size="2048"/>
				<biases offset="236964" size="256"/>
			</blobs>
		</layer>
		<layer id="42" name="460" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="461/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="461/Broadcast/9786" precision="FP32" type="Tile">
			<data axis="3" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="461" precision="FP32" type="Eltwise">
			<data operation="mul"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="462" precision="FP32" type="Eltwise">
			<data operation="sum"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="463" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="237220" size="32768"/>
				<biases offset="269988" size="512"/>
			</blobs>
		</layer>
		<layer id="48" name="465" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="270500" size="4"/>
			</blobs>
		</layer>
		<layer id="49" name="466" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="270504" size="4608"/>
				<biases offset="275112" size="512"/>
			</blobs>
		</layer>
		<layer id="50" name="468" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="275624" size="4"/>
			</blobs>
		</layer>
		<layer id="51" name="469" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="275628" size="32768"/>
				<biases offset="308396" size="256"/>
			</blobs>
		</layer>
		<layer id="52" name="472" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="32,32" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="32,32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="473" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="8" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="308652" size="2048"/>
				<biases offset="310700" size="32"/>
			</blobs>
		</layer>
		<layer id="54" name="474" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="310732" size="4"/>
			</blobs>
		</layer>
		<layer id="55" name="475" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
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
				<weights offset="310736" size="2048"/>
				<biases offset="312784" size="256"/>
			</blobs>
		</layer>
		<layer id="56" name="476" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="477/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="477/Broadcast/9790" precision="FP32" type="Tile">
			<data axis="3" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="477" precision="FP32" type="Eltwise">
			<data operation="mul"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="478" precision="FP32" type="Eltwise">
			<data operation="sum"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="479" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="313040" size="32768"/>
				<biases offset="345808" size="512"/>
			</blobs>
		</layer>
		<layer id="62" name="481" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="346320" size="4"/>
			</blobs>
		</layer>
		<layer id="63" name="482" precision="FP32" type="Convolution">
			<data dilations="1,1" group="128" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="346324" size="4608"/>
				<biases offset="350932" size="512"/>
			</blobs>
		</layer>
		<layer id="64" name="484" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="351444" size="4"/>
			</blobs>
		</layer>
		<layer id="65" name="485" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="351448" size="32768"/>
				<biases offset="384216" size="256"/>
			</blobs>
		</layer>
		<layer id="66" name="488" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="32,32" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="32,32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="489" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="8" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="384472" size="2048"/>
				<biases offset="386520" size="32"/>
			</blobs>
		</layer>
		<layer id="68" name="490" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="386552" size="4"/>
			</blobs>
		</layer>
		<layer id="69" name="491" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
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
				<weights offset="386556" size="2048"/>
				<biases offset="388604" size="256"/>
			</blobs>
		</layer>
		<layer id="70" name="492" precision="FP32" type="Activation">
			<data type="sigmoid"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="493/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="493/Broadcast/9782" precision="FP32" type="Tile">
			<data axis="3" tiles="32"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="493" precision="FP32" type="Eltwise">
			<data operation="mul"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="494" precision="FP32" type="Eltwise">
			<data operation="sum"/>
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
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="495" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="388860" size="65536"/>
				<biases offset="454396" size="1024"/>
			</blobs>
		</layer>
		<layer id="76" name="497" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="455420" size="4"/>
			</blobs>
		</layer>
		<layer id="77" name="498" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="455424" size="9216"/>
				<biases offset="464640" size="1024"/>
			</blobs>
		</layer>
		<layer id="78" name="500" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="465664" size="4"/>
			</blobs>
		</layer>
		<layer id="79" name="501" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="465668" size="131072"/>
				<biases offset="596740" size="512"/>
			</blobs>
		</layer>
		<layer id="80" name="504" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="505" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="597252" size="8192"/>
				<biases offset="605444" size="64"/>
			</blobs>
		</layer>
		<layer id="82" name="506" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="605508" size="4"/>
			</blobs>
		</layer>
		<layer id="83" name="507" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="605512" size="8192"/>
				<biases offset="613704" size="512"/>
			</blobs>
		</layer>
		<layer id="84" name="508" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="509/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="509/Broadcast/9754" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="509" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="510" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="614216" size="131072"/>
				<biases offset="745288" size="1024"/>
			</blobs>
		</layer>
		<layer id="89" name="512" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="746312" size="4"/>
			</blobs>
		</layer>
		<layer id="90" name="513" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="746316" size="9216"/>
				<biases offset="755532" size="1024"/>
			</blobs>
		</layer>
		<layer id="91" name="515" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="756556" size="4"/>
			</blobs>
		</layer>
		<layer id="92" name="516" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="756560" size="131072"/>
				<biases offset="887632" size="512"/>
			</blobs>
		</layer>
		<layer id="93" name="519" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="520" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="888144" size="8192"/>
				<biases offset="896336" size="64"/>
			</blobs>
		</layer>
		<layer id="95" name="521" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="896400" size="4"/>
			</blobs>
		</layer>
		<layer id="96" name="522" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="896404" size="8192"/>
				<biases offset="904596" size="512"/>
			</blobs>
		</layer>
		<layer id="97" name="523" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="524/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="524/Broadcast/9774" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="524" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="525" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="526" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="905108" size="131072"/>
				<biases offset="1036180" size="1024"/>
			</blobs>
		</layer>
		<layer id="103" name="528" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1037204" size="4"/>
			</blobs>
		</layer>
		<layer id="104" name="529" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1037208" size="9216"/>
				<biases offset="1046424" size="1024"/>
			</blobs>
		</layer>
		<layer id="105" name="531" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1047448" size="4"/>
			</blobs>
		</layer>
		<layer id="106" name="532" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1047452" size="131072"/>
				<biases offset="1178524" size="512"/>
			</blobs>
		</layer>
		<layer id="107" name="535" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="536" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1179036" size="8192"/>
				<biases offset="1187228" size="64"/>
			</blobs>
		</layer>
		<layer id="109" name="537" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1187292" size="4"/>
			</blobs>
		</layer>
		<layer id="110" name="538" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="1187296" size="8192"/>
				<biases offset="1195488" size="512"/>
			</blobs>
		</layer>
		<layer id="111" name="539" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="540/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="540/Broadcast/9794" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="540" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="541" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="542" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1196000" size="131072"/>
				<biases offset="1327072" size="1024"/>
			</blobs>
		</layer>
		<layer id="117" name="544" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1328096" size="4"/>
			</blobs>
		</layer>
		<layer id="118" name="545" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1328100" size="9216"/>
				<biases offset="1337316" size="1024"/>
			</blobs>
		</layer>
		<layer id="119" name="547" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1338340" size="4"/>
			</blobs>
		</layer>
		<layer id="120" name="548" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1338344" size="131072"/>
				<biases offset="1469416" size="512"/>
			</blobs>
		</layer>
		<layer id="121" name="551" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="552" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1469928" size="8192"/>
				<biases offset="1478120" size="64"/>
			</blobs>
		</layer>
		<layer id="123" name="553" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1478184" size="4"/>
			</blobs>
		</layer>
		<layer id="124" name="554" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="1478188" size="8192"/>
				<biases offset="1486380" size="512"/>
			</blobs>
		</layer>
		<layer id="125" name="555" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="556/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="556/Broadcast/9802" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="556" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="557" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="558" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1486892" size="131072"/>
				<biases offset="1617964" size="1024"/>
			</blobs>
		</layer>
		<layer id="131" name="560" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1618988" size="4"/>
			</blobs>
		</layer>
		<layer id="132" name="561" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1618992" size="9216"/>
				<biases offset="1628208" size="1024"/>
			</blobs>
		</layer>
		<layer id="133" name="563" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1629232" size="4"/>
			</blobs>
		</layer>
		<layer id="134" name="564" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1629236" size="131072"/>
				<biases offset="1760308" size="512"/>
			</blobs>
		</layer>
		<layer id="135" name="567" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="568" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1760820" size="8192"/>
				<biases offset="1769012" size="64"/>
			</blobs>
		</layer>
		<layer id="137" name="569" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1769076" size="4"/>
			</blobs>
		</layer>
		<layer id="138" name="570" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="1769080" size="8192"/>
				<biases offset="1777272" size="512"/>
			</blobs>
		</layer>
		<layer id="139" name="571" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="572/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="572/Broadcast/9766" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="572" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="573" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="574" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1777784" size="131072"/>
				<biases offset="1908856" size="1024"/>
			</blobs>
		</layer>
		<layer id="145" name="576" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1909880" size="4"/>
			</blobs>
		</layer>
		<layer id="146" name="577" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1909884" size="9216"/>
				<biases offset="1919100" size="1024"/>
			</blobs>
		</layer>
		<layer id="147" name="579" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1920124" size="4"/>
			</blobs>
		</layer>
		<layer id="148" name="580" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1920128" size="131072"/>
				<biases offset="2051200" size="512"/>
			</blobs>
		</layer>
		<layer id="149" name="583" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="584" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2051712" size="8192"/>
				<biases offset="2059904" size="64"/>
			</blobs>
		</layer>
		<layer id="151" name="585" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2059968" size="4"/>
			</blobs>
		</layer>
		<layer id="152" name="586" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="2059972" size="8192"/>
				<biases offset="2068164" size="512"/>
			</blobs>
		</layer>
		<layer id="153" name="587" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="588/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="588/Broadcast/9778" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="588" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="589" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="590" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2068676" size="131072"/>
				<biases offset="2199748" size="1024"/>
			</blobs>
		</layer>
		<layer id="159" name="592" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2200772" size="4"/>
			</blobs>
		</layer>
		<layer id="160" name="593" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2200776" size="9216"/>
				<biases offset="2209992" size="1024"/>
			</blobs>
		</layer>
		<layer id="161" name="595" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2211016" size="4"/>
			</blobs>
		</layer>
		<layer id="162" name="596" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2211020" size="131072"/>
				<biases offset="2342092" size="512"/>
			</blobs>
		</layer>
		<layer id="163" name="599" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="16,16" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="16,16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="600" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2342604" size="8192"/>
				<biases offset="2350796" size="64"/>
			</blobs>
		</layer>
		<layer id="165" name="601" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2350860" size="4"/>
			</blobs>
		</layer>
		<layer id="166" name="602" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="2350864" size="8192"/>
				<biases offset="2359056" size="512"/>
			</blobs>
		</layer>
		<layer id="167" name="603" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="604/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="16"/>
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
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="604/Broadcast/9758" precision="FP32" type="Tile">
			<data axis="3" tiles="16"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="604" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="605" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="606" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="512" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2359568" size="262144"/>
				<biases offset="2621712" size="2048"/>
			</blobs>
		</layer>
		<layer id="173" name="608" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2623760" size="4"/>
			</blobs>
		</layer>
		<layer id="174" name="609" precision="FP32" type="Convolution">
			<data dilations="1,1" group="512" kernel="3,3" output="512" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2623764" size="18432"/>
				<biases offset="2642196" size="2048"/>
			</blobs>
		</layer>
		<layer id="175" name="611" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2644244" size="4"/>
			</blobs>
		</layer>
		<layer id="176" name="612" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2644248" size="262144"/>
				<biases offset="2906392" size="512"/>
			</blobs>
		</layer>
		<layer id="177" name="615" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="8,8" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="8,8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="616" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2906904" size="8192"/>
				<biases offset="2915096" size="64"/>
			</blobs>
		</layer>
		<layer id="179" name="617" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2915160" size="4"/>
			</blobs>
		</layer>
		<layer id="180" name="618" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="2915164" size="8192"/>
				<biases offset="2923356" size="512"/>
			</blobs>
		</layer>
		<layer id="181" name="619" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="620/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="8"/>
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
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="620/Broadcast/9762" precision="FP32" type="Tile">
			<data axis="3" tiles="8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="620" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="621" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2923868" size="131072"/>
				<biases offset="3054940" size="1024"/>
			</blobs>
		</layer>
		<layer id="186" name="623" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3055964" size="4"/>
			</blobs>
		</layer>
		<layer id="187" name="624" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3055968" size="9216"/>
				<biases offset="3065184" size="1024"/>
			</blobs>
		</layer>
		<layer id="188" name="626" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3066208" size="4"/>
			</blobs>
		</layer>
		<layer id="189" name="627" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3066212" size="131072"/>
				<biases offset="3197284" size="512"/>
			</blobs>
		</layer>
		<layer id="190" name="630" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="8,8" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="8,8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="631" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3197796" size="8192"/>
				<biases offset="3205988" size="64"/>
			</blobs>
		</layer>
		<layer id="192" name="632" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3206052" size="4"/>
			</blobs>
		</layer>
		<layer id="193" name="633" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="3206056" size="8192"/>
				<biases offset="3214248" size="512"/>
			</blobs>
		</layer>
		<layer id="194" name="634" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="635/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="8"/>
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
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="635/Broadcast/9770" precision="FP32" type="Tile">
			<data axis="3" tiles="8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="635" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="636" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="637" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3214760" size="131072"/>
				<biases offset="3345832" size="1024"/>
			</blobs>
		</layer>
		<layer id="200" name="639" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3346856" size="4"/>
			</blobs>
		</layer>
		<layer id="201" name="640" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3346860" size="9216"/>
				<biases offset="3356076" size="1024"/>
			</blobs>
		</layer>
		<layer id="202" name="642" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3357100" size="4"/>
			</blobs>
		</layer>
		<layer id="203" name="643" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3357104" size="131072"/>
				<biases offset="3488176" size="512"/>
			</blobs>
		</layer>
		<layer id="204" name="646" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="8,8" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="floor" strides="8,8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="647" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
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
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3488688" size="8192"/>
				<biases offset="3496880" size="64"/>
			</blobs>
		</layer>
		<layer id="206" name="648" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3496944" size="4"/>
			</blobs>
		</layer>
		<layer id="207" name="649" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
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
				<weights offset="3496948" size="8192"/>
				<biases offset="3505140" size="512"/>
			</blobs>
		</layer>
		<layer id="208" name="650" precision="FP32" type="Activation">
			<data type="sigmoid"/>
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
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="651/Broadcast/" precision="FP32" type="Tile">
			<data axis="2" tiles="8"/>
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
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="651/Broadcast/9810" precision="FP32" type="Tile">
			<data axis="3" tiles="8"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="651" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="212" name="652" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="653" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="512" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3505652" size="262144"/>
				<biases offset="3767796" size="2048"/>
			</blobs>
		</layer>
		<layer id="214" name="655" precision="FP32" type="PReLU">
			<data channel_shared="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3769844" size="4"/>
			</blobs>
		</layer>
		<layer id="215" name="656" precision="FP32" type="Convolution">
			<data dilations="1,1" group="512" kernel="8,8" output="512" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3769848" size="131072"/>
				<biases offset="3900920" size="2048"/>
			</blobs>
		</layer>
		<layer id="216" name="658" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
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
				<weights offset="3902968" size="524288"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="2" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="3" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="2" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="1" to-layer="17" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="18" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="1"/>
		<edge from-layer="18" from-port="2" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="3" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="2" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="3" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="2" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="3" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="31" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="31" to-port="1"/>
		<edge from-layer="18" from-port="2" to-layer="32" to-port="0"/>
		<edge from-layer="31" from-port="2" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="3" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="2" to-layer="35" to-port="0"/>
		<edge from-layer="35" from-port="3" to-layer="36" to-port="0"/>
		<edge from-layer="36" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="37" from-port="3" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="3" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="2" to-layer="41" to-port="0"/>
		<edge from-layer="41" from-port="3" to-layer="42" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="44" to-port="0"/>
		<edge from-layer="37" from-port="3" to-layer="45" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="45" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="46" to-port="0"/>
		<edge from-layer="45" from-port="2" to-layer="46" to-port="1"/>
		<edge from-layer="46" from-port="2" to-layer="47" to-port="0"/>
		<edge from-layer="47" from-port="3" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="49" from-port="3" to-layer="50" to-port="0"/>
		<edge from-layer="50" from-port="2" to-layer="51" to-port="0"/>
		<edge from-layer="51" from-port="3" to-layer="52" to-port="0"/>
		<edge from-layer="52" from-port="1" to-layer="53" to-port="0"/>
		<edge from-layer="53" from-port="3" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="2" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="3" to-layer="56" to-port="0"/>
		<edge from-layer="56" from-port="1" to-layer="57" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="58" to-port="0"/>
		<edge from-layer="51" from-port="3" to-layer="59" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="1"/>
		<edge from-layer="46" from-port="2" to-layer="60" to-port="0"/>
		<edge from-layer="59" from-port="2" to-layer="60" to-port="1"/>
		<edge from-layer="60" from-port="2" to-layer="61" to-port="0"/>
		<edge from-layer="61" from-port="3" to-layer="62" to-port="0"/>
		<edge from-layer="62" from-port="2" to-layer="63" to-port="0"/>
		<edge from-layer="63" from-port="3" to-layer="64" to-port="0"/>
		<edge from-layer="64" from-port="2" to-layer="65" to-port="0"/>
		<edge from-layer="65" from-port="3" to-layer="66" to-port="0"/>
		<edge from-layer="66" from-port="1" to-layer="67" to-port="0"/>
		<edge from-layer="67" from-port="3" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="2" to-layer="69" to-port="0"/>
		<edge from-layer="69" from-port="3" to-layer="70" to-port="0"/>
		<edge from-layer="70" from-port="1" to-layer="71" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="72" to-port="0"/>
		<edge from-layer="65" from-port="3" to-layer="73" to-port="0"/>
		<edge from-layer="72" from-port="1" to-layer="73" to-port="1"/>
		<edge from-layer="60" from-port="2" to-layer="74" to-port="0"/>
		<edge from-layer="73" from-port="2" to-layer="74" to-port="1"/>
		<edge from-layer="74" from-port="2" to-layer="75" to-port="0"/>
		<edge from-layer="75" from-port="3" to-layer="76" to-port="0"/>
		<edge from-layer="76" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="77" from-port="3" to-layer="78" to-port="0"/>
		<edge from-layer="78" from-port="2" to-layer="79" to-port="0"/>
		<edge from-layer="79" from-port="3" to-layer="80" to-port="0"/>
		<edge from-layer="80" from-port="1" to-layer="81" to-port="0"/>
		<edge from-layer="81" from-port="3" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="2" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="3" to-layer="84" to-port="0"/>
		<edge from-layer="84" from-port="1" to-layer="85" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="86" to-port="0"/>
		<edge from-layer="79" from-port="3" to-layer="87" to-port="0"/>
		<edge from-layer="86" from-port="1" to-layer="87" to-port="1"/>
		<edge from-layer="87" from-port="2" to-layer="88" to-port="0"/>
		<edge from-layer="88" from-port="3" to-layer="89" to-port="0"/>
		<edge from-layer="89" from-port="2" to-layer="90" to-port="0"/>
		<edge from-layer="90" from-port="3" to-layer="91" to-port="0"/>
		<edge from-layer="91" from-port="2" to-layer="92" to-port="0"/>
		<edge from-layer="92" from-port="3" to-layer="93" to-port="0"/>
		<edge from-layer="93" from-port="1" to-layer="94" to-port="0"/>
		<edge from-layer="94" from-port="3" to-layer="95" to-port="0"/>
		<edge from-layer="95" from-port="2" to-layer="96" to-port="0"/>
		<edge from-layer="96" from-port="3" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="99" to-port="0"/>
		<edge from-layer="92" from-port="3" to-layer="100" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="100" to-port="1"/>
		<edge from-layer="87" from-port="2" to-layer="101" to-port="0"/>
		<edge from-layer="100" from-port="2" to-layer="101" to-port="1"/>
		<edge from-layer="101" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="3" to-layer="103" to-port="0"/>
		<edge from-layer="103" from-port="2" to-layer="104" to-port="0"/>
		<edge from-layer="104" from-port="3" to-layer="105" to-port="0"/>
		<edge from-layer="105" from-port="2" to-layer="106" to-port="0"/>
		<edge from-layer="106" from-port="3" to-layer="107" to-port="0"/>
		<edge from-layer="107" from-port="1" to-layer="108" to-port="0"/>
		<edge from-layer="108" from-port="3" to-layer="109" to-port="0"/>
		<edge from-layer="109" from-port="2" to-layer="110" to-port="0"/>
		<edge from-layer="110" from-port="3" to-layer="111" to-port="0"/>
		<edge from-layer="111" from-port="1" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="113" to-port="0"/>
		<edge from-layer="106" from-port="3" to-layer="114" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="114" to-port="1"/>
		<edge from-layer="101" from-port="2" to-layer="115" to-port="0"/>
		<edge from-layer="114" from-port="2" to-layer="115" to-port="1"/>
		<edge from-layer="115" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="116" from-port="3" to-layer="117" to-port="0"/>
		<edge from-layer="117" from-port="2" to-layer="118" to-port="0"/>
		<edge from-layer="118" from-port="3" to-layer="119" to-port="0"/>
		<edge from-layer="119" from-port="2" to-layer="120" to-port="0"/>
		<edge from-layer="120" from-port="3" to-layer="121" to-port="0"/>
		<edge from-layer="121" from-port="1" to-layer="122" to-port="0"/>
		<edge from-layer="122" from-port="3" to-layer="123" to-port="0"/>
		<edge from-layer="123" from-port="2" to-layer="124" to-port="0"/>
		<edge from-layer="124" from-port="3" to-layer="125" to-port="0"/>
		<edge from-layer="125" from-port="1" to-layer="126" to-port="0"/>
		<edge from-layer="126" from-port="1" to-layer="127" to-port="0"/>
		<edge from-layer="120" from-port="3" to-layer="128" to-port="0"/>
		<edge from-layer="127" from-port="1" to-layer="128" to-port="1"/>
		<edge from-layer="115" from-port="2" to-layer="129" to-port="0"/>
		<edge from-layer="128" from-port="2" to-layer="129" to-port="1"/>
		<edge from-layer="129" from-port="2" to-layer="130" to-port="0"/>
		<edge from-layer="130" from-port="3" to-layer="131" to-port="0"/>
		<edge from-layer="131" from-port="2" to-layer="132" to-port="0"/>
		<edge from-layer="132" from-port="3" to-layer="133" to-port="0"/>
		<edge from-layer="133" from-port="2" to-layer="134" to-port="0"/>
		<edge from-layer="134" from-port="3" to-layer="135" to-port="0"/>
		<edge from-layer="135" from-port="1" to-layer="136" to-port="0"/>
		<edge from-layer="136" from-port="3" to-layer="137" to-port="0"/>
		<edge from-layer="137" from-port="2" to-layer="138" to-port="0"/>
		<edge from-layer="138" from-port="3" to-layer="139" to-port="0"/>
		<edge from-layer="139" from-port="1" to-layer="140" to-port="0"/>
		<edge from-layer="140" from-port="1" to-layer="141" to-port="0"/>
		<edge from-layer="134" from-port="3" to-layer="142" to-port="0"/>
		<edge from-layer="141" from-port="1" to-layer="142" to-port="1"/>
		<edge from-layer="129" from-port="2" to-layer="143" to-port="0"/>
		<edge from-layer="142" from-port="2" to-layer="143" to-port="1"/>
		<edge from-layer="143" from-port="2" to-layer="144" to-port="0"/>
		<edge from-layer="144" from-port="3" to-layer="145" to-port="0"/>
		<edge from-layer="145" from-port="2" to-layer="146" to-port="0"/>
		<edge from-layer="146" from-port="3" to-layer="147" to-port="0"/>
		<edge from-layer="147" from-port="2" to-layer="148" to-port="0"/>
		<edge from-layer="148" from-port="3" to-layer="149" to-port="0"/>
		<edge from-layer="149" from-port="1" to-layer="150" to-port="0"/>
		<edge from-layer="150" from-port="3" to-layer="151" to-port="0"/>
		<edge from-layer="151" from-port="2" to-layer="152" to-port="0"/>
		<edge from-layer="152" from-port="3" to-layer="153" to-port="0"/>
		<edge from-layer="153" from-port="1" to-layer="154" to-port="0"/>
		<edge from-layer="154" from-port="1" to-layer="155" to-port="0"/>
		<edge from-layer="148" from-port="3" to-layer="156" to-port="0"/>
		<edge from-layer="155" from-port="1" to-layer="156" to-port="1"/>
		<edge from-layer="143" from-port="2" to-layer="157" to-port="0"/>
		<edge from-layer="156" from-port="2" to-layer="157" to-port="1"/>
		<edge from-layer="157" from-port="2" to-layer="158" to-port="0"/>
		<edge from-layer="158" from-port="3" to-layer="159" to-port="0"/>
		<edge from-layer="159" from-port="2" to-layer="160" to-port="0"/>
		<edge from-layer="160" from-port="3" to-layer="161" to-port="0"/>
		<edge from-layer="161" from-port="2" to-layer="162" to-port="0"/>
		<edge from-layer="162" from-port="3" to-layer="163" to-port="0"/>
		<edge from-layer="163" from-port="1" to-layer="164" to-port="0"/>
		<edge from-layer="164" from-port="3" to-layer="165" to-port="0"/>
		<edge from-layer="165" from-port="2" to-layer="166" to-port="0"/>
		<edge from-layer="166" from-port="3" to-layer="167" to-port="0"/>
		<edge from-layer="167" from-port="1" to-layer="168" to-port="0"/>
		<edge from-layer="168" from-port="1" to-layer="169" to-port="0"/>
		<edge from-layer="162" from-port="3" to-layer="170" to-port="0"/>
		<edge from-layer="169" from-port="1" to-layer="170" to-port="1"/>
		<edge from-layer="157" from-port="2" to-layer="171" to-port="0"/>
		<edge from-layer="170" from-port="2" to-layer="171" to-port="1"/>
		<edge from-layer="171" from-port="2" to-layer="172" to-port="0"/>
		<edge from-layer="172" from-port="3" to-layer="173" to-port="0"/>
		<edge from-layer="173" from-port="2" to-layer="174" to-port="0"/>
		<edge from-layer="174" from-port="3" to-layer="175" to-port="0"/>
		<edge from-layer="175" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="176" from-port="3" to-layer="177" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="178" to-port="0"/>
		<edge from-layer="178" from-port="3" to-layer="179" to-port="0"/>
		<edge from-layer="179" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="180" from-port="3" to-layer="181" to-port="0"/>
		<edge from-layer="181" from-port="1" to-layer="182" to-port="0"/>
		<edge from-layer="182" from-port="1" to-layer="183" to-port="0"/>
		<edge from-layer="176" from-port="3" to-layer="184" to-port="0"/>
		<edge from-layer="183" from-port="1" to-layer="184" to-port="1"/>
		<edge from-layer="184" from-port="2" to-layer="185" to-port="0"/>
		<edge from-layer="185" from-port="3" to-layer="186" to-port="0"/>
		<edge from-layer="186" from-port="2" to-layer="187" to-port="0"/>
		<edge from-layer="187" from-port="3" to-layer="188" to-port="0"/>
		<edge from-layer="188" from-port="2" to-layer="189" to-port="0"/>
		<edge from-layer="189" from-port="3" to-layer="190" to-port="0"/>
		<edge from-layer="190" from-port="1" to-layer="191" to-port="0"/>
		<edge from-layer="191" from-port="3" to-layer="192" to-port="0"/>
		<edge from-layer="192" from-port="2" to-layer="193" to-port="0"/>
		<edge from-layer="193" from-port="3" to-layer="194" to-port="0"/>
		<edge from-layer="194" from-port="1" to-layer="195" to-port="0"/>
		<edge from-layer="195" from-port="1" to-layer="196" to-port="0"/>
		<edge from-layer="189" from-port="3" to-layer="197" to-port="0"/>
		<edge from-layer="196" from-port="1" to-layer="197" to-port="1"/>
		<edge from-layer="184" from-port="2" to-layer="198" to-port="0"/>
		<edge from-layer="197" from-port="2" to-layer="198" to-port="1"/>
		<edge from-layer="198" from-port="2" to-layer="199" to-port="0"/>
		<edge from-layer="199" from-port="3" to-layer="200" to-port="0"/>
		<edge from-layer="200" from-port="2" to-layer="201" to-port="0"/>
		<edge from-layer="201" from-port="3" to-layer="202" to-port="0"/>
		<edge from-layer="202" from-port="2" to-layer="203" to-port="0"/>
		<edge from-layer="203" from-port="3" to-layer="204" to-port="0"/>
		<edge from-layer="204" from-port="1" to-layer="205" to-port="0"/>
		<edge from-layer="205" from-port="3" to-layer="206" to-port="0"/>
		<edge from-layer="206" from-port="2" to-layer="207" to-port="0"/>
		<edge from-layer="207" from-port="3" to-layer="208" to-port="0"/>
		<edge from-layer="208" from-port="1" to-layer="209" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="210" to-port="0"/>
		<edge from-layer="203" from-port="3" to-layer="211" to-port="0"/>
		<edge from-layer="210" from-port="1" to-layer="211" to-port="1"/>
		<edge from-layer="198" from-port="2" to-layer="212" to-port="0"/>
		<edge from-layer="211" from-port="2" to-layer="212" to-port="1"/>
		<edge from-layer="212" from-port="2" to-layer="213" to-port="0"/>
		<edge from-layer="213" from-port="3" to-layer="214" to-port="0"/>
		<edge from-layer="214" from-port="2" to-layer="215" to-port="0"/>
		<edge from-layer="215" from-port="3" to-layer="216" to-port="0"/>
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
			<input_model value="DIR/Mobilenet_se_focal_121000.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,128,128]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'0': {'scale': array([254.99991075]), 'mean': None}}"/>
			<mean_values value="()"/>
			<model_name value="face-reidentification-retail-0095"/>
			<move_to_preprocess value="False"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'0': array([  1,   3, 128, 128])}"/>
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
