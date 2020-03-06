<?xml version="1.0" ?>
<net batch="1" name="icv-pedestrian-detection-mobilenet-ssd-v2.0" version="6">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Mul_/Fused_Mul_/FusedScaleShift_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="32" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="3456"/>
				<biases offset="3480" size="128"/>
			</blobs>
		</layer>
		<layer id="3" name="relu1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv2_1/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="32" kernel="3,3" output="32" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3608" size="1152"/>
				<biases offset="4760" size="128"/>
			</blobs>
		</layer>
		<layer id="5" name="relu2_1/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="conv2_1/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="56" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4888" size="7168"/>
				<biases offset="12056" size="224"/>
			</blobs>
		</layer>
		<layer id="7" name="relu2_1/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="conv2_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="56" kernel="3,3" output="56" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>192</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
			<blobs>
				<weights offset="12280" size="2016"/>
				<biases offset="14296" size="224"/>
			</blobs>
		</layer>
		<layer id="9" name="relu2_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv2_2/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="112" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>56</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
			<blobs>
				<weights offset="14520" size="25088"/>
				<biases offset="39608" size="448"/>
			</blobs>
		</layer>
		<layer id="11" name="relu2_2/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="conv3_1/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="112" kernel="3,3" output="112" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
			<blobs>
				<weights offset="40056" size="4032"/>
				<biases offset="44088" size="448"/>
			</blobs>
		</layer>
		<layer id="13" name="relu3_1/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="conv3_1/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>112</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
			<blobs>
				<weights offset="44536" size="43008"/>
				<biases offset="87544" size="384"/>
			</blobs>
		</layer>
		<layer id="15" name="relu3_1/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="conv3_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="96" kernel="3,3" output="96" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>96</dim>
					<dim>168</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
			<blobs>
				<weights offset="87928" size="3456"/>
				<biases offset="91384" size="384"/>
			</blobs>
		</layer>
		<layer id="17" name="relu3_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="conv3_2/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="232" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
			<blobs>
				<weights offset="91768" size="89088"/>
				<biases offset="180856" size="928"/>
			</blobs>
		</layer>
		<layer id="19" name="relu3_2/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="conv4_1/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="232" kernel="3,3" output="232" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
			<blobs>
				<weights offset="181784" size="8352"/>
				<biases offset="190136" size="928"/>
			</blobs>
		</layer>
		<layer id="21" name="relu4_1/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="conv4_1/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="208" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>232</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
			<blobs>
				<weights offset="191064" size="193024"/>
				<biases offset="384088" size="832"/>
			</blobs>
		</layer>
		<layer id="23" name="relu4_1/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="conv4_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="208" kernel="3,3" output="208" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>48</dim>
					<dim>84</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>208</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="384920" size="7488"/>
				<biases offset="392408" size="832"/>
			</blobs>
		</layer>
		<layer id="25" name="relu4_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>208</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="conv4_2/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="352" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>208</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="393240" size="292864"/>
				<biases offset="686104" size="1408"/>
			</blobs>
		</layer>
		<layer id="27" name="relu4_2/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="conv5_1/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="352" kernel="3,3" output="352" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="687512" size="12672"/>
				<biases offset="700184" size="1408"/>
			</blobs>
		</layer>
		<layer id="29" name="relu5_1/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="conv5_1/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="320" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>352</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="701592" size="450560"/>
				<biases offset="1152152" size="1280"/>
			</blobs>
		</layer>
		<layer id="31" name="relu5_1/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="conv5_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="320" kernel="3,3" output="320" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1153432" size="11520"/>
				<biases offset="1164952" size="1280"/>
			</blobs>
		</layer>
		<layer id="33" name="relu5_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="conv5_2/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="296" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1166232" size="378880"/>
				<biases offset="1545112" size="1184"/>
			</blobs>
		</layer>
		<layer id="35" name="relu5_2/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="conv5_3/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="296" kernel="3,3" output="296" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1546296" size="10656"/>
				<biases offset="1556952" size="1184"/>
			</blobs>
		</layer>
		<layer id="37" name="relu5_3/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="conv5_3/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="248" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>296</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1558136" size="293632"/>
				<biases offset="1851768" size="992"/>
			</blobs>
		</layer>
		<layer id="39" name="relu5_3/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="conv5_4/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="248" kernel="3,3" output="248" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1852760" size="8928"/>
				<biases offset="1861688" size="992"/>
			</blobs>
		</layer>
		<layer id="41" name="relu5_4/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv5_4/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="272" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>248</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1862680" size="269824"/>
				<biases offset="2132504" size="1088"/>
			</blobs>
		</layer>
		<layer id="43" name="relu5_4/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="conv5_5/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="272" kernel="3,3" output="272" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2133592" size="9792"/>
				<biases offset="2143384" size="1088"/>
			</blobs>
		</layer>
		<layer id="45" name="relu5_5/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="conv5_5/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="176" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>272</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2144472" size="191488"/>
				<biases offset="2335960" size="704"/>
			</blobs>
		</layer>
		<layer id="47" name="relu5_5/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="conv4_3_0_norm_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="16" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2336664" size="101376"/>
				<biases offset="2438040" size="64"/>
			</blobs>
		</layer>
		<layer id="49" name="conv4_3_0_norm_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="conv4_3_0_norm_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="conv4_3_norm_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="16" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2438104" size="101376"/>
				<biases offset="2539480" size="64"/>
			</blobs>
		</layer>
		<layer id="52" name="conv4_3_norm_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="conv4_3_norm_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="conv5_6/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="176" kernel="3,3" output="176" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>176</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2539544" size="6336"/>
				<biases offset="2545880" size="704"/>
			</blobs>
		</layer>
		<layer id="55" name="relu5_6/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>176</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="conv5_6/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2546584" size="180224"/>
				<biases offset="2726808" size="1024"/>
			</blobs>
		</layer>
		<layer id="57" name="relu5_6/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="conv6/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="256" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2727832" size="9216"/>
				<biases offset="2737048" size="1024"/>
			</blobs>
		</layer>
		<layer id="59" name="relu6/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="conv6/sep" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2738072" size="98304"/>
				<biases offset="2836376" size="384"/>
			</blobs>
		</layer>
		<layer id="61" name="relu6/sep" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="fc7_0_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2836760" size="82944"/>
				<biases offset="2919704" size="96"/>
			</blobs>
		</layer>
		<layer id="63" name="fc7_0_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="fc7_0_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="fc7_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2919800" size="82944"/>
				<biases offset="3002744" size="96"/>
			</blobs>
		</layer>
		<layer id="66" name="fc7_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="fc7_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="conv6_1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="96" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3002840" size="36864"/>
				<biases offset="3039704" size="384"/>
			</blobs>
		</layer>
		<layer id="69" name="relu6_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="conv6_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="96" kernel="3,3" output="96" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3040088" size="3456"/>
				<biases offset="3043544" size="384"/>
			</blobs>
		</layer>
		<layer id="71" name="relu6_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="conv6_2_new" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="120" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3043928" size="46080"/>
				<biases offset="3090008" size="480"/>
			</blobs>
		</layer>
		<layer id="73" name="relu6_2" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="conv6_2_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3090488" size="103680"/>
				<biases offset="3194168" size="96"/>
			</blobs>
		</layer>
		<layer id="75" name="conv6_2_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="conv6_2_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="conv6_2_mbox_loc_bigpriors" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3194264" size="103680"/>
				<biases offset="3297944" size="96"/>
			</blobs>
		</layer>
		<layer id="78" name="conv6_2_mbox_loc_perm_bigpriors" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="conv6_2_mbox_loc_flat_bigpriors" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="conv7_1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="120" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3298040" size="57600"/>
				<biases offset="3355640" size="480"/>
			</blobs>
		</layer>
		<layer id="81" name="relu7_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="conv7_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="120" kernel="3,3" output="120" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3356120" size="4320"/>
				<biases offset="3360440" size="480"/>
			</blobs>
		</layer>
		<layer id="83" name="relu7_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="conv7_2_new" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="144" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3360920" size="69120"/>
				<biases offset="3430040" size="576"/>
			</blobs>
		</layer>
		<layer id="85" name="relu7_2" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="conv7_2_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3430616" size="124416"/>
				<biases offset="3555032" size="96"/>
			</blobs>
		</layer>
		<layer id="87" name="conv7_2_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="conv7_2_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="conv7_2_mbox_loc_bigpriors" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="24" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3555128" size="124416"/>
				<biases offset="3679544" size="96"/>
			</blobs>
		</layer>
		<layer id="90" name="conv7_2_mbox_loc_perm_bigpriors" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="conv7_2_mbox_loc_flat_bigpriors" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>24</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="conv8_1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="120" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3679640" size="69120"/>
				<biases offset="3748760" size="480"/>
			</blobs>
		</layer>
		<layer id="93" name="relu8_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="conv8_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="120" kernel="3,3" output="120" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3749240" size="4320"/>
				<biases offset="3753560" size="480"/>
			</blobs>
		</layer>
		<layer id="95" name="relu8_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="conv8_2_new" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="216" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3754040" size="103680"/>
				<biases offset="3857720" size="864"/>
			</blobs>
		</layer>
		<layer id="97" name="relu8_2" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="conv8_2_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="16" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3858584" size="124416"/>
				<biases offset="3983000" size="64"/>
			</blobs>
		</layer>
		<layer id="99" name="conv8_2_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="conv8_2_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="conv9_1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3983064" size="55296"/>
				<biases offset="4038360" size="256"/>
			</blobs>
		</layer>
		<layer id="102" name="relu9_1" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="conv9_2/dw" precision="FP32" type="Convolution">
			<data dilations="1,1" group="64" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4038616" size="2304"/>
				<biases offset="4040920" size="256"/>
			</blobs>
		</layer>
		<layer id="104" name="relu9_2/dw" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="conv9_2_new" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4041176" size="32768"/>
				<biases offset="4073944" size="512"/>
			</blobs>
		</layer>
		<layer id="106" name="relu9_2" precision="FP32" type="ReLU">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="conv9_2_mbox_loc" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="16" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4074456" size="73728"/>
				<biases offset="4148184" size="64"/>
			</blobs>
		</layer>
		<layer id="108" name="conv9_2_mbox_loc_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="conv9_2_mbox_loc_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="mbox_loc" precision="FP32" type="Concat">
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
					<dim>1584</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>432</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>432</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>96</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="10">
					<dim>1</dim>
					<dim>48512</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="conv4_3_0_norm_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="8" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4148248" size="50688"/>
				<biases offset="4198936" size="32"/>
			</blobs>
		</layer>
		<layer id="112" name="conv4_3_0_norm_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="conv4_3_0_norm_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="conv4_3_norm_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="8" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4198968" size="50688"/>
				<biases offset="4249656" size="32"/>
			</blobs>
		</layer>
		<layer id="115" name="conv4_3_norm_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="conv4_3_norm_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24</dim>
					<dim>42</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8064</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="fc7_0_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4249688" size="41472"/>
				<biases offset="4291160" size="48"/>
			</blobs>
		</layer>
		<layer id="118" name="fc7_0_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="fc7_0_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="fc7_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4291208" size="41472"/>
				<biases offset="4332680" size="48"/>
			</blobs>
		</layer>
		<layer id="121" name="fc7_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="fc7_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>21</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3024</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="conv6_2_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4332728" size="51840"/>
				<biases offset="4384568" size="48"/>
			</blobs>
		</layer>
		<layer id="124" name="conv6_2_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="conv6_2_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>792</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="conv6_2_mbox_conf_bigpriors" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4384616" size="51840"/>
				<biases offset="4436456" size="48"/>
			</blobs>
		</layer>
		<layer id="127" name="conv6_2_mbox_conf_perm_bigpriors" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="conv6_2_mbox_conf_flat_bigpriors" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>6</dim>
					<dim>11</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>792</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="conv7_2_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4436504" size="62208"/>
				<biases offset="4498712" size="48"/>
			</blobs>
		</layer>
		<layer id="130" name="conv7_2_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="conv7_2_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>216</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="conv7_2_mbox_conf_bigpriors" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="12" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4498760" size="62208"/>
				<biases offset="4560968" size="48"/>
			</blobs>
		</layer>
		<layer id="133" name="conv7_2_mbox_conf_perm_bigpriors" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="conv7_2_mbox_conf_flat_bigpriors" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>6</dim>
					<dim>12</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>216</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="conv8_2_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="8" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4561016" size="62208"/>
				<biases offset="4623224" size="32"/>
			</blobs>
		</layer>
		<layer id="136" name="conv8_2_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="conv8_2_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>3</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>48</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="conv9_2_mbox_conf" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="8" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4623256" size="36864"/>
				<biases offset="4660120" size="32"/>
			</blobs>
		</layer>
		<layer id="139" name="conv9_2_mbox_conf_perm" precision="FP32" type="Permute">
			<data order="0,2,3,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="conv9_2_mbox_conf_flat" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="mbox_conf" precision="FP32" type="Concat">
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
					<dim>792</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>216</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>216</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>48</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="10">
					<dim>1</dim>
					<dim>24256</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="236/Output_0/Data__const" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4660152" size="12"/>
			</blobs>
		</layer>
		<layer id="143" name="mbox_conf_reshape" precision="FP32" type="Reshape">
			<data axis="0" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>24256</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>12128</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="mbox_conf_softmax" precision="FP32" type="SoftMax">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12128</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12128</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="mbox_conf_flatten" precision="FP32" type="Flatten">
			<data axis="1" end_axis="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12128</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>24256</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="conv4_3_0_norm_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="38.400001525878906" min_size="16.0" offset="0.5" step="16.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="conv4_3_norm_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="76.80000305175781" min_size="38.400001525878906" offset="0.5" step="16.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>176</dim>
					<dim>24</dim>
					<dim>42</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>16128</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="fc7_0_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="124.80000305175781" min_size="76.80000305175781" offset="0.5" step="32.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="fc7_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="172.8000030517578" min_size="124.80000305175781" offset="0.5" step="32.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>12</dim>
					<dim>21</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>6048</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="conv6_2_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="201.60000610351562" min_size="172.8000030517578" offset="0.5" step="64.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="conv6_2_mbox_priorbox_bigpriors" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="230.39999389648438" min_size="201.60000610351562" offset="0.5" step="64.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>120</dim>
					<dim>6</dim>
					<dim>11</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1584</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="conv7_2_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="259.20001220703125" min_size="230.39999389648438" offset="0.5" step="128.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="conv7_2_mbox_priorbox_bigpriors" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0,3.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="288.0" min_size="259.20001220703125" offset="0.5" step="128.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>144</dim>
					<dim>3</dim>
					<dim>6</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="conv8_2_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="345.6000061035156" min_size="288.0" offset="0.5" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>216</dim>
					<dim>2</dim>
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="conv9_2_mbox_priorbox" precision="FP32" type="PriorBox">
			<data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="403.20001220703125" min_size="345.6000061035156" offset="0.5" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>384</dim>
					<dim>672</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="mbox_priorbox" precision="FP32" type="Concat">
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
					<dim>1584</dim>
				</port>
				<port id="6">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
				<port id="7">
					<dim>1</dim>
					<dim>2</dim>
					<dim>432</dim>
				</port>
				<port id="8">
					<dim>1</dim>
					<dim>2</dim>
					<dim>96</dim>
				</port>
				<port id="9">
					<dim>1</dim>
					<dim>2</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="10">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48512</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="detection_out" precision="FP32" type="DetectionOutput">
			<data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.009999999776482582" eta="1.0" input_height="1" input_width="1" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="1" num_classes="2" share_location="1" top_k="400" variance_encoded_in_target="0" visualize="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48512</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>24256</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48512</dim>
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
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="3" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="3" to-layer="13" to-port="0"/>
		<edge from-layer="13" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="14" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="3" to-layer="19" to-port="0"/>
		<edge from-layer="19" from-port="1" to-layer="20" to-port="0"/>
		<edge from-layer="20" from-port="3" to-layer="21" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="3" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="1" to-layer="24" to-port="0"/>
		<edge from-layer="24" from-port="3" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="26" to-port="0"/>
		<edge from-layer="26" from-port="3" to-layer="27" to-port="0"/>
		<edge from-layer="27" from-port="1" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="3" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="1" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="3" to-layer="31" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="3" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="3" to-layer="35" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="0"/>
		<edge from-layer="36" from-port="3" to-layer="37" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="3" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="40" to-port="0"/>
		<edge from-layer="40" from-port="3" to-layer="41" to-port="0"/>
		<edge from-layer="41" from-port="1" to-layer="42" to-port="0"/>
		<edge from-layer="42" from-port="3" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="44" to-port="0"/>
		<edge from-layer="44" from-port="3" to-layer="45" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="46" to-port="0"/>
		<edge from-layer="46" from-port="3" to-layer="47" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="3" to-layer="49" to-port="0"/>
		<edge from-layer="49" from-port="1" to-layer="50" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="51" to-port="0"/>
		<edge from-layer="51" from-port="3" to-layer="52" to-port="0"/>
		<edge from-layer="52" from-port="1" to-layer="53" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="3" to-layer="55" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="56" to-port="0"/>
		<edge from-layer="56" from-port="3" to-layer="57" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="58" to-port="0"/>
		<edge from-layer="58" from-port="3" to-layer="59" to-port="0"/>
		<edge from-layer="59" from-port="1" to-layer="60" to-port="0"/>
		<edge from-layer="60" from-port="3" to-layer="61" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="62" to-port="0"/>
		<edge from-layer="62" from-port="3" to-layer="63" to-port="0"/>
		<edge from-layer="63" from-port="1" to-layer="64" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="65" to-port="0"/>
		<edge from-layer="65" from-port="3" to-layer="66" to-port="0"/>
		<edge from-layer="66" from-port="1" to-layer="67" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="3" to-layer="69" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="70" to-port="0"/>
		<edge from-layer="70" from-port="3" to-layer="71" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="72" to-port="0"/>
		<edge from-layer="72" from-port="3" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="74" to-port="0"/>
		<edge from-layer="74" from-port="3" to-layer="75" to-port="0"/>
		<edge from-layer="75" from-port="1" to-layer="76" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="77" to-port="0"/>
		<edge from-layer="77" from-port="3" to-layer="78" to-port="0"/>
		<edge from-layer="78" from-port="1" to-layer="79" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="80" to-port="0"/>
		<edge from-layer="80" from-port="3" to-layer="81" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="82" to-port="0"/>
		<edge from-layer="82" from-port="3" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="1" to-layer="84" to-port="0"/>
		<edge from-layer="84" from-port="3" to-layer="85" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="86" to-port="0"/>
		<edge from-layer="86" from-port="3" to-layer="87" to-port="0"/>
		<edge from-layer="87" from-port="1" to-layer="88" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="89" to-port="0"/>
		<edge from-layer="89" from-port="3" to-layer="90" to-port="0"/>
		<edge from-layer="90" from-port="1" to-layer="91" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="92" to-port="0"/>
		<edge from-layer="92" from-port="3" to-layer="93" to-port="0"/>
		<edge from-layer="93" from-port="1" to-layer="94" to-port="0"/>
		<edge from-layer="94" from-port="3" to-layer="95" to-port="0"/>
		<edge from-layer="95" from-port="1" to-layer="96" to-port="0"/>
		<edge from-layer="96" from-port="3" to-layer="97" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="3" to-layer="99" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="100" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="101" to-port="0"/>
		<edge from-layer="101" from-port="3" to-layer="102" to-port="0"/>
		<edge from-layer="102" from-port="1" to-layer="103" to-port="0"/>
		<edge from-layer="103" from-port="3" to-layer="104" to-port="0"/>
		<edge from-layer="104" from-port="1" to-layer="105" to-port="0"/>
		<edge from-layer="105" from-port="3" to-layer="106" to-port="0"/>
		<edge from-layer="106" from-port="1" to-layer="107" to-port="0"/>
		<edge from-layer="107" from-port="3" to-layer="108" to-port="0"/>
		<edge from-layer="108" from-port="1" to-layer="109" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="110" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="110" to-port="1"/>
		<edge from-layer="64" from-port="1" to-layer="110" to-port="2"/>
		<edge from-layer="67" from-port="1" to-layer="110" to-port="3"/>
		<edge from-layer="76" from-port="1" to-layer="110" to-port="4"/>
		<edge from-layer="79" from-port="1" to-layer="110" to-port="5"/>
		<edge from-layer="88" from-port="1" to-layer="110" to-port="6"/>
		<edge from-layer="91" from-port="1" to-layer="110" to-port="7"/>
		<edge from-layer="100" from-port="1" to-layer="110" to-port="8"/>
		<edge from-layer="109" from-port="1" to-layer="110" to-port="9"/>
		<edge from-layer="47" from-port="1" to-layer="111" to-port="0"/>
		<edge from-layer="111" from-port="3" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="113" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="114" to-port="0"/>
		<edge from-layer="114" from-port="3" to-layer="115" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="116" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="117" to-port="0"/>
		<edge from-layer="117" from-port="3" to-layer="118" to-port="0"/>
		<edge from-layer="118" from-port="1" to-layer="119" to-port="0"/>
		<edge from-layer="61" from-port="1" to-layer="120" to-port="0"/>
		<edge from-layer="120" from-port="3" to-layer="121" to-port="0"/>
		<edge from-layer="121" from-port="1" to-layer="122" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="123" to-port="0"/>
		<edge from-layer="123" from-port="3" to-layer="124" to-port="0"/>
		<edge from-layer="124" from-port="1" to-layer="125" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="126" to-port="0"/>
		<edge from-layer="126" from-port="3" to-layer="127" to-port="0"/>
		<edge from-layer="127" from-port="1" to-layer="128" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="129" to-port="0"/>
		<edge from-layer="129" from-port="3" to-layer="130" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="131" to-port="0"/>
		<edge from-layer="85" from-port="1" to-layer="132" to-port="0"/>
		<edge from-layer="132" from-port="3" to-layer="133" to-port="0"/>
		<edge from-layer="133" from-port="1" to-layer="134" to-port="0"/>
		<edge from-layer="97" from-port="1" to-layer="135" to-port="0"/>
		<edge from-layer="135" from-port="3" to-layer="136" to-port="0"/>
		<edge from-layer="136" from-port="1" to-layer="137" to-port="0"/>
		<edge from-layer="106" from-port="1" to-layer="138" to-port="0"/>
		<edge from-layer="138" from-port="3" to-layer="139" to-port="0"/>
		<edge from-layer="139" from-port="1" to-layer="140" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="141" to-port="0"/>
		<edge from-layer="116" from-port="1" to-layer="141" to-port="1"/>
		<edge from-layer="119" from-port="1" to-layer="141" to-port="2"/>
		<edge from-layer="122" from-port="1" to-layer="141" to-port="3"/>
		<edge from-layer="125" from-port="1" to-layer="141" to-port="4"/>
		<edge from-layer="128" from-port="1" to-layer="141" to-port="5"/>
		<edge from-layer="131" from-port="1" to-layer="141" to-port="6"/>
		<edge from-layer="134" from-port="1" to-layer="141" to-port="7"/>
		<edge from-layer="137" from-port="1" to-layer="141" to-port="8"/>
		<edge from-layer="140" from-port="1" to-layer="141" to-port="9"/>
		<edge from-layer="141" from-port="10" to-layer="143" to-port="0"/>
		<edge from-layer="142" from-port="1" to-layer="143" to-port="1"/>
		<edge from-layer="143" from-port="2" to-layer="144" to-port="0"/>
		<edge from-layer="144" from-port="1" to-layer="145" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="146" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="146" to-port="1"/>
		<edge from-layer="47" from-port="1" to-layer="147" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="147" to-port="1"/>
		<edge from-layer="61" from-port="1" to-layer="148" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="148" to-port="1"/>
		<edge from-layer="61" from-port="1" to-layer="149" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="149" to-port="1"/>
		<edge from-layer="73" from-port="1" to-layer="150" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="150" to-port="1"/>
		<edge from-layer="73" from-port="1" to-layer="151" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="151" to-port="1"/>
		<edge from-layer="85" from-port="1" to-layer="152" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="152" to-port="1"/>
		<edge from-layer="85" from-port="1" to-layer="153" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="153" to-port="1"/>
		<edge from-layer="97" from-port="1" to-layer="154" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="154" to-port="1"/>
		<edge from-layer="106" from-port="1" to-layer="155" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="155" to-port="1"/>
		<edge from-layer="146" from-port="2" to-layer="156" to-port="0"/>
		<edge from-layer="147" from-port="2" to-layer="156" to-port="1"/>
		<edge from-layer="148" from-port="2" to-layer="156" to-port="2"/>
		<edge from-layer="149" from-port="2" to-layer="156" to-port="3"/>
		<edge from-layer="150" from-port="2" to-layer="156" to-port="4"/>
		<edge from-layer="151" from-port="2" to-layer="156" to-port="5"/>
		<edge from-layer="152" from-port="2" to-layer="156" to-port="6"/>
		<edge from-layer="153" from-port="2" to-layer="156" to-port="7"/>
		<edge from-layer="154" from-port="2" to-layer="156" to-port="8"/>
		<edge from-layer="155" from-port="2" to-layer="156" to-port="9"/>
		<edge from-layer="110" from-port="10" to-layer="157" to-port="0"/>
		<edge from-layer="145" from-port="1" to-layer="157" to-port="1"/>
		<edge from-layer="156" from-port="10" to-layer="157" to-port="2"/>
	</edges>
	<meta_data>
		<MO_version value="2019.3.0-227-g3a4f9de20"/>
		<cli_parameters>
			<blobs_as_inputs value="False"/>
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
			<generate_experimental_IR_V10 value="False"/>
			<input value="data"/>
			<input_model value="DIR/icv-pedestrian-detection-mobilenet-ssd-v2.0.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/icv-pedestrian-detection-mobilenet-ssd-v2.0.prototxt"/>
			<input_shape value="[1,3,384,672]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="False"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'scale': array([58.82352941]), 'mean': array([104., 117., 123.])}}"/>
			<mean_values value="data[104.0,117.0,123.0]"/>
			<model_name value="pedestrian-detection-adas-0002"/>
			<move_to_preprocess value="False"/>
			<output value="['detection_out']"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'data': array([  1,   3, 384, 672])}"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="data[58.8235294117647]"/>
			<silent value="False"/>
			<steps value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, generate_deprecated_IR_V2, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
