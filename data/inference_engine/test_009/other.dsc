<?xml version="1.0" ?>
<net batch="1" name="PVANet + R-FCN" version="4">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>544</dim>
					<dim>992</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="ScaleShift/Add_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>544</dim>
					<dim>992</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>544</dim>
					<dim>992</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="7,7" output="16" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>544</dim>
					<dim>992</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="9408"/>
				<biases offset="9432" size="64"/>
			</blobs>
		</layer>
		<layer id="3" name="conv1_1/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv1_1/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="conv1_1/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9496" size="128"/>
				<biases offset="9624" size="128"/>
			</blobs>
		</layer>
		<layer id="6" name="conv1_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="pool1" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>272</dim>
					<dim>496</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="conv2_1/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9752" size="3072"/>
				<biases offset="12824" size="96"/>
			</blobs>
		</layer>
		<layer id="9" name="conv2_1/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv2_1/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12920" size="20736"/>
				<biases offset="33656" size="96"/>
			</blobs>
		</layer>
		<layer id="11" name="conv2_1/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="conv2_1/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="conv2_1/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="33752" size="192"/>
				<biases offset="33944" size="192"/>
			</blobs>
		</layer>
		<layer id="14" name="conv2_1/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="conv2_1/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="34136" size="12288"/>
				<biases offset="46424" size="256"/>
			</blobs>
		</layer>
		<layer id="16" name="conv2_1/proj" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="46680" size="8192"/>
				<biases offset="54872" size="256"/>
			</blobs>
		</layer>
		<layer id="17" name="conv2_1" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="conv2_2/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="55128" size="6144"/>
				<biases offset="61272" size="96"/>
			</blobs>
		</layer>
		<layer id="19" name="conv2_2/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="conv2_2/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="61368" size="20736"/>
				<biases offset="82104" size="96"/>
			</blobs>
		</layer>
		<layer id="21" name="conv2_2/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="conv2_2/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="conv2_2/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="82200" size="192"/>
				<biases offset="82392" size="192"/>
			</blobs>
		</layer>
		<layer id="24" name="conv2_2/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="conv2_2/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="82584" size="12288"/>
				<biases offset="94872" size="256"/>
			</blobs>
		</layer>
		<layer id="26" name="conv2_2" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="conv2_3/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="95128" size="6144"/>
				<biases offset="101272" size="96"/>
			</blobs>
		</layer>
		<layer id="28" name="conv2_3/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="conv2_3/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
			<blobs>
				<weights offset="101368" size="20736"/>
				<biases offset="122104" size="96"/>
			</blobs>
		</layer>
		<layer id="30" name="conv2_3/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="conv2_3/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="Pooling_" precision="FP32" type="Pooling">
			<data kernel="1,1" pads_begin="0,0" pads_end="0,0" pool-method="max" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="conv2_3/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="122200" size="192"/>
				<biases offset="122392" size="192"/>
			</blobs>
		</layer>
		<layer id="34" name="conv2_3/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="conv2_3/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="122584" size="12288"/>
				<biases offset="134872" size="256"/>
			</blobs>
		</layer>
		<layer id="36" name="Pooling_4086" precision="FP32" type="Pooling">
			<data kernel="1,1" pads_begin="0,0" pads_end="0,0" pool-method="max" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>136</dim>
					<dim>248</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="conv2_3" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="conv3_1/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="135128" size="12288"/>
				<biases offset="147416" size="192"/>
			</blobs>
		</layer>
		<layer id="39" name="conv3_1/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="conv3_1/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="147608" size="82944"/>
				<biases offset="230552" size="192"/>
			</blobs>
		</layer>
		<layer id="41" name="conv3_1/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv3_1/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="conv3_1/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="230744" size="384"/>
				<biases offset="231128" size="384"/>
			</blobs>
		</layer>
		<layer id="44" name="conv3_1/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="conv3_1/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="231512" size="49152"/>
				<biases offset="280664" size="512"/>
			</blobs>
		</layer>
		<layer id="46" name="conv3_1/proj" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="281176" size="32768"/>
				<biases offset="313944" size="512"/>
			</blobs>
		</layer>
		<layer id="47" name="conv3_1" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="conv3_2/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="314456" size="24576"/>
				<biases offset="339032" size="192"/>
			</blobs>
		</layer>
		<layer id="49" name="conv3_2/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="conv3_2/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="339224" size="82944"/>
				<biases offset="422168" size="192"/>
			</blobs>
		</layer>
		<layer id="51" name="conv3_2/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="conv3_2/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="conv3_2/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="422360" size="384"/>
				<biases offset="422744" size="384"/>
			</blobs>
		</layer>
		<layer id="54" name="conv3_2/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="conv3_2/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="423128" size="49152"/>
				<biases offset="472280" size="512"/>
			</blobs>
		</layer>
		<layer id="56" name="conv3_2" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="conv3_3/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="472792" size="24576"/>
				<biases offset="497368" size="192"/>
			</blobs>
		</layer>
		<layer id="58" name="conv3_3/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="conv3_3/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="497560" size="82944"/>
				<biases offset="580504" size="192"/>
			</blobs>
		</layer>
		<layer id="60" name="conv3_3/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="conv3_3/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="conv3_3/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="580696" size="384"/>
				<biases offset="581080" size="384"/>
			</blobs>
		</layer>
		<layer id="63" name="conv3_3/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="conv3_3/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="581464" size="49152"/>
				<biases offset="630616" size="512"/>
			</blobs>
		</layer>
		<layer id="65" name="conv3_3" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="conv3_4/1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="631128" size="24576"/>
				<biases offset="655704" size="192"/>
			</blobs>
		</layer>
		<layer id="67" name="conv3_4/1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="conv3_4/2/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="655896" size="82944"/>
				<biases offset="738840" size="192"/>
			</blobs>
		</layer>
		<layer id="69" name="conv3_4/2/neg" precision="FP32" type="Power">
			<data power="1.0" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="conv3_4/2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="conv3_4/2/scale/Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="739032" size="384"/>
				<biases offset="739416" size="384"/>
			</blobs>
		</layer>
		<layer id="72" name="conv3_4/2/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="conv3_4/3/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
			<blobs>
				<weights offset="739800" size="49152"/>
				<biases offset="788952" size="512"/>
			</blobs>
		</layer>
		<layer id="74" name="conv3_4" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="downsample" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="conv4_1/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="789464" size="32768"/>
				<biases offset="822232" size="256"/>
			</blobs>
		</layer>
		<layer id="77" name="conv4_1/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="conv4_1/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="48" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="822488" size="24576"/>
				<biases offset="847064" size="192"/>
			</blobs>
		</layer>
		<layer id="79" name="conv4_1/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="conv4_1/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="847256" size="221184"/>
				<biases offset="1068440" size="512"/>
			</blobs>
		</layer>
		<layer id="81" name="conv4_1/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="conv4_1/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1068952" size="12288"/>
				<biases offset="1081240" size="96"/>
			</blobs>
		</layer>
		<layer id="83" name="conv4_1/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="conv4_1/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1081336" size="41472"/>
				<biases offset="1122808" size="192"/>
			</blobs>
		</layer>
		<layer id="85" name="conv4_1/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="conv4_1/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1123000" size="82944"/>
				<biases offset="1205944" size="192"/>
			</blobs>
		</layer>
		<layer id="87" name="conv4_1/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="conv4_1/incep/pool" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="conv4_1/incep/poolproj/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1206136" size="65536"/>
				<biases offset="1271672" size="512"/>
			</blobs>
		</layer>
		<layer id="90" name="conv4_1/incep/poolproj/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="conv4_1/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>368</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="conv4_1/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>368</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1272184" size="376832"/>
				<biases offset="1649016" size="1024"/>
			</blobs>
		</layer>
		<layer id="93" name="conv4_1/proj" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>68</dim>
					<dim>124</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1650040" size="131072"/>
				<biases offset="1781112" size="1024"/>
			</blobs>
		</layer>
		<layer id="94" name="conv4_1" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="conv4_2/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1782136" size="65536"/>
				<biases offset="1847672" size="256"/>
			</blobs>
		</layer>
		<layer id="96" name="conv4_2/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="conv4_2/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1847928" size="65536"/>
				<biases offset="1913464" size="256"/>
			</blobs>
		</layer>
		<layer id="98" name="conv4_2/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="conv4_2/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1913720" size="294912"/>
				<biases offset="2208632" size="512"/>
			</blobs>
		</layer>
		<layer id="100" name="conv4_2/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="conv4_2/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2209144" size="24576"/>
				<biases offset="2233720" size="96"/>
			</blobs>
		</layer>
		<layer id="102" name="conv4_2/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="conv4_2/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2233816" size="41472"/>
				<biases offset="2275288" size="192"/>
			</blobs>
		</layer>
		<layer id="104" name="conv4_2/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="conv4_2/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2275480" size="82944"/>
				<biases offset="2358424" size="192"/>
			</blobs>
		</layer>
		<layer id="106" name="conv4_2/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="conv4_2/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="conv4_2/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2358616" size="245760"/>
				<biases offset="2604376" size="1024"/>
			</blobs>
		</layer>
		<layer id="109" name="conv4_2" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="conv4_3/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2605400" size="65536"/>
				<biases offset="2670936" size="256"/>
			</blobs>
		</layer>
		<layer id="111" name="conv4_3/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="conv4_3/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2671192" size="65536"/>
				<biases offset="2736728" size="256"/>
			</blobs>
		</layer>
		<layer id="113" name="conv4_3/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="conv4_3/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2736984" size="294912"/>
				<biases offset="3031896" size="512"/>
			</blobs>
		</layer>
		<layer id="115" name="conv4_3/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="conv4_3/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3032408" size="24576"/>
				<biases offset="3056984" size="96"/>
			</blobs>
		</layer>
		<layer id="117" name="conv4_3/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="conv4_3/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3057080" size="41472"/>
				<biases offset="3098552" size="192"/>
			</blobs>
		</layer>
		<layer id="119" name="conv4_3/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="conv4_3/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3098744" size="82944"/>
				<biases offset="3181688" size="192"/>
			</blobs>
		</layer>
		<layer id="121" name="conv4_3/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="conv4_3/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="conv4_3/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3181880" size="245760"/>
				<biases offset="3427640" size="1024"/>
			</blobs>
		</layer>
		<layer id="124" name="conv4_3" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="conv4_4/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3428664" size="65536"/>
				<biases offset="3494200" size="256"/>
			</blobs>
		</layer>
		<layer id="126" name="conv4_4/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="conv4_4/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3494456" size="65536"/>
				<biases offset="3559992" size="256"/>
			</blobs>
		</layer>
		<layer id="128" name="conv4_4/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="conv4_4/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="128" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3560248" size="294912"/>
				<biases offset="3855160" size="512"/>
			</blobs>
		</layer>
		<layer id="130" name="conv4_4/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="conv4_4/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3855672" size="24576"/>
				<biases offset="3880248" size="96"/>
			</blobs>
		</layer>
		<layer id="132" name="conv4_4/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="conv4_4/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3880344" size="41472"/>
				<biases offset="3921816" size="192"/>
			</blobs>
		</layer>
		<layer id="134" name="conv4_4/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="conv4_4/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="48" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3922008" size="82944"/>
				<biases offset="4004952" size="192"/>
			</blobs>
		</layer>
		<layer id="136" name="conv4_4/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="conv4_4/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>48</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="conv4_4/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>240</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4005144" size="245760"/>
				<biases offset="4250904" size="1024"/>
			</blobs>
		</layer>
		<layer id="139" name="conv4_4" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="conv5_1/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4251928" size="65536"/>
				<biases offset="4317464" size="256"/>
			</blobs>
		</layer>
		<layer id="141" name="conv5_1/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="conv5_1/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4317720" size="98304"/>
				<biases offset="4416024" size="384"/>
			</blobs>
		</layer>
		<layer id="143" name="conv5_1/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="conv5_1/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4416408" size="663552"/>
				<biases offset="5079960" size="768"/>
			</blobs>
		</layer>
		<layer id="145" name="conv5_1/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="conv5_1/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5080728" size="32768"/>
				<biases offset="5113496" size="128"/>
			</blobs>
		</layer>
		<layer id="147" name="conv5_1/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="conv5_1/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5113624" size="73728"/>
				<biases offset="5187352" size="256"/>
			</blobs>
		</layer>
		<layer id="149" name="conv5_1/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="conv5_1/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5187608" size="147456"/>
				<biases offset="5335064" size="256"/>
			</blobs>
		</layer>
		<layer id="151" name="conv5_1/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="conv5_1/incep/pool" precision="FP32" type="Pooling">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="conv5_1/incep/poolproj/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5335320" size="131072"/>
				<biases offset="5466392" size="512"/>
			</blobs>
		</layer>
		<layer id="154" name="conv5_1/incep/poolproj/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="conv5_1/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="4">
					<dim>1</dim>
					<dim>448</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="conv5_1/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="384" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>448</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="5466904" size="688128"/>
				<biases offset="6155032" size="1536"/>
			</blobs>
		</layer>
		<layer id="157" name="conv5_1/proj" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="384" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6156568" size="393216"/>
				<biases offset="6549784" size="1536"/>
			</blobs>
		</layer>
		<layer id="158" name="conv5_1" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="159" name="conv5_2/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6551320" size="98304"/>
				<biases offset="6649624" size="256"/>
			</blobs>
		</layer>
		<layer id="160" name="conv5_2/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="conv5_2/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6649880" size="147456"/>
				<biases offset="6797336" size="384"/>
			</blobs>
		</layer>
		<layer id="162" name="conv5_2/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="conv5_2/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="6797720" size="663552"/>
				<biases offset="7461272" size="768"/>
			</blobs>
		</layer>
		<layer id="164" name="conv5_2/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="conv5_2/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7462040" size="49152"/>
				<biases offset="7511192" size="128"/>
			</blobs>
		</layer>
		<layer id="166" name="conv5_2/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="conv5_2/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7511320" size="73728"/>
				<biases offset="7585048" size="256"/>
			</blobs>
		</layer>
		<layer id="168" name="conv5_2/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="conv5_2/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7585304" size="147456"/>
				<biases offset="7732760" size="256"/>
			</blobs>
		</layer>
		<layer id="170" name="conv5_2/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="conv5_2/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="conv5_2/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="384" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7733016" size="491520"/>
				<biases offset="8224536" size="1536"/>
			</blobs>
		</layer>
		<layer id="173" name="conv5_2" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="conv5_3/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8226072" size="98304"/>
				<biases offset="8324376" size="256"/>
			</blobs>
		</layer>
		<layer id="175" name="conv5_3/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="conv5_3/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8324632" size="147456"/>
				<biases offset="8472088" size="384"/>
			</blobs>
		</layer>
		<layer id="177" name="conv5_3/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="conv5_3/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8472472" size="663552"/>
				<biases offset="9136024" size="768"/>
			</blobs>
		</layer>
		<layer id="179" name="conv5_3/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="conv5_3/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9136792" size="49152"/>
				<biases offset="9185944" size="128"/>
			</blobs>
		</layer>
		<layer id="181" name="conv5_3/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="conv5_3/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9186072" size="73728"/>
				<biases offset="9259800" size="256"/>
			</blobs>
		</layer>
		<layer id="183" name="conv5_3/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="conv5_3/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9260056" size="147456"/>
				<biases offset="9407512" size="256"/>
			</blobs>
		</layer>
		<layer id="185" name="conv5_3/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="conv5_3/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="conv5_3/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="384" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9407768" size="491520"/>
				<biases offset="9899288" size="1536"/>
			</blobs>
		</layer>
		<layer id="188" name="conv5_3" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="conv5_4/incep/0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9900824" size="98304"/>
				<biases offset="9999128" size="256"/>
			</blobs>
		</layer>
		<layer id="190" name="conv5_4/incep/0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="conv5_4/incep/1_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="9999384" size="147456"/>
				<biases offset="10146840" size="384"/>
			</blobs>
		</layer>
		<layer id="192" name="conv5_4/incep/1_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="conv5_4/incep/1_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="10147224" size="663552"/>
				<biases offset="10810776" size="768"/>
			</blobs>
		</layer>
		<layer id="194" name="conv5_4/incep/1_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="conv5_4/incep/2_reduce/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="32" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="10811544" size="49152"/>
				<biases offset="10860696" size="128"/>
			</blobs>
		</layer>
		<layer id="196" name="conv5_4/incep/2_reduce/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="conv5_4/incep/2_0/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="10860824" size="73728"/>
				<biases offset="10934552" size="256"/>
			</blobs>
		</layer>
		<layer id="198" name="conv5_4/incep/2_0/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="conv5_4/incep/2_1/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="10934808" size="147456"/>
				<biases offset="11082264" size="256"/>
			</blobs>
		</layer>
		<layer id="200" name="conv5_4/incep/2_1/relu" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="conv5_4/incep" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>192</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="conv5_4/out/conv" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="384" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11082520" size="491520"/>
				<biases offset="11574040" size="1536"/>
			</blobs>
		</layer>
		<layer id="203" name="conv5_4" precision="FP32" type="Eltwise">
			<data coeff="1.0,1.0" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="upsample" precision="FP32" type="Deconvolution">
			<data dilations="1,1" group="384" kernel="4,4" output="384" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>384</dim>
					<dim>17</dim>
					<dim>31</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11575576" size="24576"/>
			</blobs>
		</layer>
		<layer id="205" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>384</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>768</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="convf_rpn" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>768</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11600152" size="393216"/>
				<biases offset="11993368" size="512"/>
			</blobs>
		</layer>
		<layer id="207" name="reluf_rpn" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="convf_2" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>768</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11993880" size="393216"/>
				<biases offset="12387096" size="512"/>
			</blobs>
		</layer>
		<layer id="209" name="reluf_2" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="concat_convf" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="detector/bbox/bbox" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="392" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>392</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12387608" size="401408"/>
				<biases offset="12789016" size="1568"/>
			</blobs>
		</layer>
		<layer id="212" name="rpn_conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12790584" size="65536"/>
				<biases offset="12856120" size="512"/>
			</blobs>
		</layer>
		<layer id="213" name="rpn_relu1" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="rpn_cls_score" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="12" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12856632" size="6144"/>
				<biases offset="12862776" size="48"/>
			</blobs>
		</layer>
		<layer id="215" name="rpn_cls_score_reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12862824" size="16"/>
			</blobs>
		</layer>
		<layer id="216" name="rpn_cls_score_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>204</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="rpn_cls_prob" precision="FP32" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>204</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>204</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="rpn_cls_prob_reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12862840" size="16"/>
			</blobs>
		</layer>
		<layer id="219" name="rpn_cls_prob_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>204</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>12</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="rpn_bbox_pred" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="24" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12862856" size="12288"/>
				<biases offset="12875144" size="96"/>
			</blobs>
		</layer>
		<layer id="221" name="im_info" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="proposal" precision="FP32" type="Proposal">
			<data base_size="16" feat_stride="16" min_size="16" nms_thresh="0.6000000238418579" post_nms_topn="200" pre_nms_topn="6000" ratio="2.6689999103546143" scale="4.0,6.0,9.0,16.0,24.0,32.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>200</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="detector/bbox/ps_roi_pooling" precision="FP32" type="PSROIPooling">
			<data group_size="7" mode="average" output_dim="8" spatial_scale="0.0625"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>392</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>200</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>200</dim>
					<dim>8</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="detector/bbox/ave_pred" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="7,7" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="ceil" strides="7,7"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>8</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>200</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="bbox_pred_reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12875240" size="8"/>
			</blobs>
		</layer>
		<layer id="226" name="bbox_pred_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>1600</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="detector/cls/cls" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="98" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>98</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12875248" size="100352"/>
				<biases offset="12975600" size="392"/>
			</blobs>
		</layer>
		<layer id="228" name="detector/cls/ps_roi_pooling" precision="FP32" type="PSROIPooling">
			<data group_size="7" mode="average" output_dim="2" spatial_scale="0.0625"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>98</dim>
					<dim>34</dim>
					<dim>62</dim>
				</port>
				<port id="1">
					<dim>200</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>200</dim>
					<dim>2</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="detector/cls/ave_pred" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="7,7" pads_begin="0,0" pads_end="0,0" pool-method="avg" rounding_type="ceil" strides="7,7"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>2</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>200</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="cls_prob" precision="FP32" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>200</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="cls_prob_reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12875240" size="8"/>
			</blobs>
		</layer>
		<layer id="232" name="cls_prob_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>400</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="rois_reshape/DimData_const" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12975992" size="12"/>
			</blobs>
		</layer>
		<layer id="234" name="rois_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>200</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="detection_out" precision="FP32" type="DetectionOutput">
			<data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE" eta="1.0" input_height="544" input_width="992" keep_top_k="200" nms_threshold="0.4000000059604645" normalized="0" num_classes="2" share_location="0" top_k="12000" variance_encoded_in_target="1" visualize="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1600</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>400</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>200</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="12" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
		<edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="3" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="1" to-layer="15" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="15" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="1"/>
		<edge from-layer="17" from-port="2" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="3" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="3" to-layer="21" to-port="0"/>
		<edge from-layer="20" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="3" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="3" to-layer="26" to-port="0"/>
		<edge from-layer="17" from-port="2" to-layer="26" to-port="1"/>
		<edge from-layer="26" from-port="2" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="3" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="3" to-layer="30" to-port="0"/>
		<edge from-layer="29" from-port="3" to-layer="31" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="31" to-port="1"/>
		<edge from-layer="31" from-port="2" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="1" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="3" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="1" to-layer="35" to-port="0"/>
		<edge from-layer="26" from-port="2" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="3" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="1" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="3" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="3" to-layer="41" to-port="0"/>
		<edge from-layer="40" from-port="3" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="3" to-layer="44" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="45" to-port="0"/>
		<edge from-layer="37" from-port="2" to-layer="46" to-port="0"/>
		<edge from-layer="45" from-port="3" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="3" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="3" to-layer="49" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="50" to-port="0"/>
		<edge from-layer="50" from-port="3" to-layer="51" to-port="0"/>
		<edge from-layer="50" from-port="3" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="1" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="53" from-port="3" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="1" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="3" to-layer="56" to-port="0"/>
		<edge from-layer="47" from-port="2" to-layer="56" to-port="1"/>
		<edge from-layer="56" from-port="2" to-layer="57" to-port="0"/>
		<edge from-layer="57" from-port="3" to-layer="58" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="0"/>
		<edge from-layer="59" from-port="3" to-layer="60" to-port="0"/>
		<edge from-layer="59" from-port="3" to-layer="61" to-port="0"/>
		<edge from-layer="60" from-port="1" to-layer="61" to-port="1"/>
		<edge from-layer="61" from-port="2" to-layer="62" to-port="0"/>
		<edge from-layer="62" from-port="3" to-layer="63" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="0"/>
		<edge from-layer="64" from-port="3" to-layer="65" to-port="0"/>
		<edge from-layer="56" from-port="2" to-layer="65" to-port="1"/>
		<edge from-layer="65" from-port="2" to-layer="66" to-port="0"/>
		<edge from-layer="66" from-port="3" to-layer="67" to-port="0"/>
		<edge from-layer="67" from-port="1" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="3" to-layer="69" to-port="0"/>
		<edge from-layer="68" from-port="3" to-layer="70" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="71" to-port="0"/>
		<edge from-layer="71" from-port="3" to-layer="72" to-port="0"/>
		<edge from-layer="72" from-port="1" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="3" to-layer="74" to-port="0"/>
		<edge from-layer="65" from-port="2" to-layer="74" to-port="1"/>
		<edge from-layer="74" from-port="2" to-layer="75" to-port="0"/>
		<edge from-layer="74" from-port="2" to-layer="76" to-port="0"/>
		<edge from-layer="76" from-port="3" to-layer="77" to-port="0"/>
		<edge from-layer="74" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="78" from-port="3" to-layer="79" to-port="0"/>
		<edge from-layer="79" from-port="1" to-layer="80" to-port="0"/>
		<edge from-layer="80" from-port="3" to-layer="81" to-port="0"/>
		<edge from-layer="74" from-port="2" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="3" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="1" to-layer="84" to-port="0"/>
		<edge from-layer="84" from-port="3" to-layer="85" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="86" to-port="0"/>
		<edge from-layer="86" from-port="3" to-layer="87" to-port="0"/>
		<edge from-layer="74" from-port="2" to-layer="88" to-port="0"/>
		<edge from-layer="88" from-port="1" to-layer="89" to-port="0"/>
		<edge from-layer="89" from-port="3" to-layer="90" to-port="0"/>
		<edge from-layer="77" from-port="1" to-layer="91" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="91" to-port="1"/>
		<edge from-layer="87" from-port="1" to-layer="91" to-port="2"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="3"/>
		<edge from-layer="91" from-port="4" to-layer="92" to-port="0"/>
		<edge from-layer="74" from-port="2" to-layer="93" to-port="0"/>
		<edge from-layer="92" from-port="3" to-layer="94" to-port="0"/>
		<edge from-layer="93" from-port="3" to-layer="94" to-port="1"/>
		<edge from-layer="94" from-port="2" to-layer="95" to-port="0"/>
		<edge from-layer="95" from-port="3" to-layer="96" to-port="0"/>
		<edge from-layer="94" from-port="2" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="3" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="99" to-port="0"/>
		<edge from-layer="99" from-port="3" to-layer="100" to-port="0"/>
		<edge from-layer="94" from-port="2" to-layer="101" to-port="0"/>
		<edge from-layer="101" from-port="3" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="1" to-layer="103" to-port="0"/>
		<edge from-layer="103" from-port="3" to-layer="104" to-port="0"/>
		<edge from-layer="104" from-port="1" to-layer="105" to-port="0"/>
		<edge from-layer="105" from-port="3" to-layer="106" to-port="0"/>
		<edge from-layer="96" from-port="1" to-layer="107" to-port="0"/>
		<edge from-layer="100" from-port="1" to-layer="107" to-port="1"/>
		<edge from-layer="106" from-port="1" to-layer="107" to-port="2"/>
		<edge from-layer="107" from-port="3" to-layer="108" to-port="0"/>
		<edge from-layer="108" from-port="3" to-layer="109" to-port="0"/>
		<edge from-layer="94" from-port="2" to-layer="109" to-port="1"/>
		<edge from-layer="109" from-port="2" to-layer="110" to-port="0"/>
		<edge from-layer="110" from-port="3" to-layer="111" to-port="0"/>
		<edge from-layer="109" from-port="2" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="3" to-layer="113" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="114" to-port="0"/>
		<edge from-layer="114" from-port="3" to-layer="115" to-port="0"/>
		<edge from-layer="109" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="116" from-port="3" to-layer="117" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="118" to-port="0"/>
		<edge from-layer="118" from-port="3" to-layer="119" to-port="0"/>
		<edge from-layer="119" from-port="1" to-layer="120" to-port="0"/>
		<edge from-layer="120" from-port="3" to-layer="121" to-port="0"/>
		<edge from-layer="111" from-port="1" to-layer="122" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="122" to-port="1"/>
		<edge from-layer="121" from-port="1" to-layer="122" to-port="2"/>
		<edge from-layer="122" from-port="3" to-layer="123" to-port="0"/>
		<edge from-layer="123" from-port="3" to-layer="124" to-port="0"/>
		<edge from-layer="109" from-port="2" to-layer="124" to-port="1"/>
		<edge from-layer="124" from-port="2" to-layer="125" to-port="0"/>
		<edge from-layer="125" from-port="3" to-layer="126" to-port="0"/>
		<edge from-layer="124" from-port="2" to-layer="127" to-port="0"/>
		<edge from-layer="127" from-port="3" to-layer="128" to-port="0"/>
		<edge from-layer="128" from-port="1" to-layer="129" to-port="0"/>
		<edge from-layer="129" from-port="3" to-layer="130" to-port="0"/>
		<edge from-layer="124" from-port="2" to-layer="131" to-port="0"/>
		<edge from-layer="131" from-port="3" to-layer="132" to-port="0"/>
		<edge from-layer="132" from-port="1" to-layer="133" to-port="0"/>
		<edge from-layer="133" from-port="3" to-layer="134" to-port="0"/>
		<edge from-layer="134" from-port="1" to-layer="135" to-port="0"/>
		<edge from-layer="135" from-port="3" to-layer="136" to-port="0"/>
		<edge from-layer="126" from-port="1" to-layer="137" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="137" to-port="1"/>
		<edge from-layer="136" from-port="1" to-layer="137" to-port="2"/>
		<edge from-layer="137" from-port="3" to-layer="138" to-port="0"/>
		<edge from-layer="138" from-port="3" to-layer="139" to-port="0"/>
		<edge from-layer="124" from-port="2" to-layer="139" to-port="1"/>
		<edge from-layer="139" from-port="2" to-layer="140" to-port="0"/>
		<edge from-layer="140" from-port="3" to-layer="141" to-port="0"/>
		<edge from-layer="139" from-port="2" to-layer="142" to-port="0"/>
		<edge from-layer="142" from-port="3" to-layer="143" to-port="0"/>
		<edge from-layer="143" from-port="1" to-layer="144" to-port="0"/>
		<edge from-layer="144" from-port="3" to-layer="145" to-port="0"/>
		<edge from-layer="139" from-port="2" to-layer="146" to-port="0"/>
		<edge from-layer="146" from-port="3" to-layer="147" to-port="0"/>
		<edge from-layer="147" from-port="1" to-layer="148" to-port="0"/>
		<edge from-layer="148" from-port="3" to-layer="149" to-port="0"/>
		<edge from-layer="149" from-port="1" to-layer="150" to-port="0"/>
		<edge from-layer="150" from-port="3" to-layer="151" to-port="0"/>
		<edge from-layer="139" from-port="2" to-layer="152" to-port="0"/>
		<edge from-layer="152" from-port="1" to-layer="153" to-port="0"/>
		<edge from-layer="153" from-port="3" to-layer="154" to-port="0"/>
		<edge from-layer="141" from-port="1" to-layer="155" to-port="0"/>
		<edge from-layer="145" from-port="1" to-layer="155" to-port="1"/>
		<edge from-layer="151" from-port="1" to-layer="155" to-port="2"/>
		<edge from-layer="154" from-port="1" to-layer="155" to-port="3"/>
		<edge from-layer="155" from-port="4" to-layer="156" to-port="0"/>
		<edge from-layer="139" from-port="2" to-layer="157" to-port="0"/>
		<edge from-layer="156" from-port="3" to-layer="158" to-port="0"/>
		<edge from-layer="157" from-port="3" to-layer="158" to-port="1"/>
		<edge from-layer="158" from-port="2" to-layer="159" to-port="0"/>
		<edge from-layer="159" from-port="3" to-layer="160" to-port="0"/>
		<edge from-layer="158" from-port="2" to-layer="161" to-port="0"/>
		<edge from-layer="161" from-port="3" to-layer="162" to-port="0"/>
		<edge from-layer="162" from-port="1" to-layer="163" to-port="0"/>
		<edge from-layer="163" from-port="3" to-layer="164" to-port="0"/>
		<edge from-layer="158" from-port="2" to-layer="165" to-port="0"/>
		<edge from-layer="165" from-port="3" to-layer="166" to-port="0"/>
		<edge from-layer="166" from-port="1" to-layer="167" to-port="0"/>
		<edge from-layer="167" from-port="3" to-layer="168" to-port="0"/>
		<edge from-layer="168" from-port="1" to-layer="169" to-port="0"/>
		<edge from-layer="169" from-port="3" to-layer="170" to-port="0"/>
		<edge from-layer="160" from-port="1" to-layer="171" to-port="0"/>
		<edge from-layer="164" from-port="1" to-layer="171" to-port="1"/>
		<edge from-layer="170" from-port="1" to-layer="171" to-port="2"/>
		<edge from-layer="171" from-port="3" to-layer="172" to-port="0"/>
		<edge from-layer="172" from-port="3" to-layer="173" to-port="0"/>
		<edge from-layer="158" from-port="2" to-layer="173" to-port="1"/>
		<edge from-layer="173" from-port="2" to-layer="174" to-port="0"/>
		<edge from-layer="174" from-port="3" to-layer="175" to-port="0"/>
		<edge from-layer="173" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="176" from-port="3" to-layer="177" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="178" to-port="0"/>
		<edge from-layer="178" from-port="3" to-layer="179" to-port="0"/>
		<edge from-layer="173" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="180" from-port="3" to-layer="181" to-port="0"/>
		<edge from-layer="181" from-port="1" to-layer="182" to-port="0"/>
		<edge from-layer="182" from-port="3" to-layer="183" to-port="0"/>
		<edge from-layer="183" from-port="1" to-layer="184" to-port="0"/>
		<edge from-layer="184" from-port="3" to-layer="185" to-port="0"/>
		<edge from-layer="175" from-port="1" to-layer="186" to-port="0"/>
		<edge from-layer="179" from-port="1" to-layer="186" to-port="1"/>
		<edge from-layer="185" from-port="1" to-layer="186" to-port="2"/>
		<edge from-layer="186" from-port="3" to-layer="187" to-port="0"/>
		<edge from-layer="187" from-port="3" to-layer="188" to-port="0"/>
		<edge from-layer="173" from-port="2" to-layer="188" to-port="1"/>
		<edge from-layer="188" from-port="2" to-layer="189" to-port="0"/>
		<edge from-layer="189" from-port="3" to-layer="190" to-port="0"/>
		<edge from-layer="188" from-port="2" to-layer="191" to-port="0"/>
		<edge from-layer="191" from-port="3" to-layer="192" to-port="0"/>
		<edge from-layer="192" from-port="1" to-layer="193" to-port="0"/>
		<edge from-layer="193" from-port="3" to-layer="194" to-port="0"/>
		<edge from-layer="188" from-port="2" to-layer="195" to-port="0"/>
		<edge from-layer="195" from-port="3" to-layer="196" to-port="0"/>
		<edge from-layer="196" from-port="1" to-layer="197" to-port="0"/>
		<edge from-layer="197" from-port="3" to-layer="198" to-port="0"/>
		<edge from-layer="198" from-port="1" to-layer="199" to-port="0"/>
		<edge from-layer="199" from-port="3" to-layer="200" to-port="0"/>
		<edge from-layer="190" from-port="1" to-layer="201" to-port="0"/>
		<edge from-layer="194" from-port="1" to-layer="201" to-port="1"/>
		<edge from-layer="200" from-port="1" to-layer="201" to-port="2"/>
		<edge from-layer="201" from-port="3" to-layer="202" to-port="0"/>
		<edge from-layer="202" from-port="3" to-layer="203" to-port="0"/>
		<edge from-layer="188" from-port="2" to-layer="203" to-port="1"/>
		<edge from-layer="203" from-port="2" to-layer="204" to-port="0"/>
		<edge from-layer="75" from-port="1" to-layer="205" to-port="0"/>
		<edge from-layer="139" from-port="2" to-layer="205" to-port="1"/>
		<edge from-layer="204" from-port="2" to-layer="205" to-port="2"/>
		<edge from-layer="205" from-port="3" to-layer="206" to-port="0"/>
		<edge from-layer="206" from-port="3" to-layer="207" to-port="0"/>
		<edge from-layer="205" from-port="3" to-layer="208" to-port="0"/>
		<edge from-layer="208" from-port="3" to-layer="209" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="210" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="210" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="211" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="212" to-port="0"/>
		<edge from-layer="212" from-port="3" to-layer="213" to-port="0"/>
		<edge from-layer="213" from-port="1" to-layer="214" to-port="0"/>
		<edge from-layer="214" from-port="3" to-layer="216" to-port="0"/>
		<edge from-layer="215" from-port="1" to-layer="216" to-port="1"/>
		<edge from-layer="216" from-port="2" to-layer="217" to-port="0"/>
		<edge from-layer="217" from-port="1" to-layer="219" to-port="0"/>
		<edge from-layer="218" from-port="1" to-layer="219" to-port="1"/>
		<edge from-layer="213" from-port="1" to-layer="220" to-port="0"/>
		<edge from-layer="219" from-port="2" to-layer="222" to-port="0"/>
		<edge from-layer="220" from-port="3" to-layer="222" to-port="1"/>
		<edge from-layer="221" from-port="0" to-layer="222" to-port="2"/>
		<edge from-layer="211" from-port="3" to-layer="223" to-port="0"/>
		<edge from-layer="222" from-port="3" to-layer="223" to-port="1"/>
		<edge from-layer="223" from-port="2" to-layer="224" to-port="0"/>
		<edge from-layer="224" from-port="1" to-layer="226" to-port="0"/>
		<edge from-layer="225" from-port="1" to-layer="226" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="227" to-port="0"/>
		<edge from-layer="227" from-port="3" to-layer="228" to-port="0"/>
		<edge from-layer="222" from-port="3" to-layer="228" to-port="1"/>
		<edge from-layer="228" from-port="2" to-layer="229" to-port="0"/>
		<edge from-layer="229" from-port="1" to-layer="230" to-port="0"/>
		<edge from-layer="230" from-port="1" to-layer="232" to-port="0"/>
		<edge from-layer="231" from-port="1" to-layer="232" to-port="1"/>
		<edge from-layer="222" from-port="3" to-layer="234" to-port="0"/>
		<edge from-layer="233" from-port="1" to-layer="234" to-port="1"/>
		<edge from-layer="226" from-port="2" to-layer="235" to-port="0"/>
		<edge from-layer="232" from-port="2" to-layer="235" to-port="1"/>
		<edge from-layer="234" from-port="2" to-layer="235" to-port="2"/>
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
			<input value="data,im_info"/>
			<input_model value="DIR/cnn.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/cnn.prototxt"/>
			<input_shape value="[1,3,544,992],[1,6]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'mean': array([102.9801, 115.9465, 122.7717]), 'scale': None}}"/>
			<mean_values value="data[102.9801,115.9465,122.7717]"/>
			<model_name value="person-detection-retail-0002"/>
			<move_to_preprocess value="False"/>
			<output value="['detection_out']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'im_info': array([1, 6]), 'data': array([  1,   3, 544, 992])}"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="()"/>
			<silent value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, finegrain_fusing, freeze_placeholder_with_value, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
