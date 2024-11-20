<?xml version="1.0"?>
<net name="torch-jit-export" version="11">
	<layers>
		<layer id="0" name="image" type="Parameter" version="opset1">
			<data shape="1,3,256,256" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="image">
					<dim>1</dim>
					<dim>3</dim>
					<dim>256</dim>
					<dim>256</dim>
					<rt_info>
						<attribute name="layout" version="0" layout="[N,C,H,W]" />
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="1" name="Constant_9851" type="Const" version="opset1">
			<data element_type="f32" shape="1, 1, 1, 1" offset="0" size="4" />
			<rt_info>
				<attribute name="preprocessing" version="0" />
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Divide_1793" type="Multiply" version="opset1">
			<data auto_broadcast="numpy" />
			<rt_info>
				<attribute name="preprocessing" version="0" />
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Gather_9850" type="Const" version="opset1">
			<data element_type="f32" shape="32, 3, 3, 3" offset="4" size="3456" />
			<rt_info>
				<attribute name="preprocessing" version="0" />
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Multiply_8742" type="Convolution" version="opset1">
			<data strides="2, 2" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Constant_8747" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="3460" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="BatchNormalization_1" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="351">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Clip_2" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="352">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="Multiply_9156" type="Const" version="opset1">
			<data element_type="f32" shape="32, 32, 1, 1" offset="3588" size="4096" />
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Multiply_8749" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="Constant_8754" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="7684" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="BatchNormalization_4" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="354">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="Clip_5" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="355">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Multiply_9162" type="Const" version="opset1">
			<data element_type="f32" shape="32, 1, 1, 3, 3" offset="7812" size="1152" />
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="Multiply_8756" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="356">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="Constant_8761" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="8964" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="BatchNormalization_7" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="357">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="Clip_8" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="358">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="Multiply_9167" type="Const" version="opset1">
			<data element_type="f32" shape="16, 32, 1, 1" offset="9092" size="2048" />
			<output>
				<port id="0" precision="FP32">
					<dim>16</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="Multiply_8763" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="Constant_8768" type="Const" version="opset1">
			<data element_type="f32" shape="1, 16, 1, 1" offset="11140" size="64" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="BatchNormalization_10" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="360">
					<dim>1</dim>
					<dim>16</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="Multiply_9173" type="Const" version="opset1">
			<data element_type="f32" shape="96, 16, 1, 1" offset="11204" size="6144" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="Multiply_8770" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="Constant_8775" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="17348" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="BatchNormalization_12" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="362">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="Clip_13" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="363">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="Multiply_9179" type="Const" version="opset1">
			<data element_type="f32" shape="96, 1, 1, 3, 3" offset="17732" size="3456" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="Multiply_8777" type="GroupConvolution" version="opset1">
			<data strides="2, 2" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="364">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="Constant_8782" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="21188" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="BatchNormalization_15" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="365">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="Clip_16" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="366">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="Multiply_9184" type="Const" version="opset1">
			<data element_type="f32" shape="24, 96, 1, 1" offset="21572" size="9216" />
			<output>
				<port id="0" precision="FP32">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="Multiply_8784" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="Constant_8789" type="Const" version="opset1">
			<data element_type="f32" shape="1, 24, 1, 1" offset="30788" size="96" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="BatchNormalization_18" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="368">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="Multiply_9190" type="Const" version="opset1">
			<data element_type="f32" shape="144, 24, 1, 1" offset="30884" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="Multiply_8791" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="Constant_8796" type="Const" version="opset1">
			<data element_type="f32" shape="1, 144, 1, 1" offset="44708" size="576" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="BatchNormalization_20" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="370">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="Clip_21" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="371">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="Multiply_9196" type="Const" version="opset1">
			<data element_type="f32" shape="144, 1, 1, 3, 3" offset="45284" size="5184" />
			<output>
				<port id="0" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="Multiply_8798" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="372">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="Constant_8803" type="Const" version="opset1">
			<data element_type="f32" shape="1, 144, 1, 1" offset="50468" size="576" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="BatchNormalization_23" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="373">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="Clip_24" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="374">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="Multiply_9201" type="Const" version="opset1">
			<data element_type="f32" shape="24, 144, 1, 1" offset="51044" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="Multiply_8805" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="Constant_8810" type="Const" version="opset1">
			<data element_type="f32" shape="1, 24, 1, 1" offset="64868" size="96" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="BatchNormalization_26" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="376">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="Add_27" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="377">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="Multiply_9207" type="Const" version="opset1">
			<data element_type="f32" shape="144, 24, 1, 1" offset="64964" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="Multiply_8812" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="Constant_8817" type="Const" version="opset1">
			<data element_type="f32" shape="1, 144, 1, 1" offset="78788" size="576" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="BatchNormalization_29" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="379">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="Clip_30" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="380">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="Multiply_9213" type="Const" version="opset1">
			<data element_type="f32" shape="144, 1, 1, 3, 3" offset="79364" size="5184" />
			<output>
				<port id="0" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="Multiply_8819" type="GroupConvolution" version="opset1">
			<data strides="2, 2" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="381">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="Constant_8824" type="Const" version="opset1">
			<data element_type="f32" shape="1, 144, 1, 1" offset="84548" size="576" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="BatchNormalization_32" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="382">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="60" name="Clip_33" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="383">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="Multiply_9218" type="Const" version="opset1">
			<data element_type="f32" shape="32, 144, 1, 1" offset="85124" size="18432" />
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="62" name="Multiply_8826" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="Constant_8831" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="103556" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="64" name="BatchNormalization_35" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="385">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="Multiply_9224" type="Const" version="opset1">
			<data element_type="f32" shape="192, 32, 1, 1" offset="103684" size="24576" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="66" name="Multiply_8833" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="Constant_8838" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="128260" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="68" name="BatchNormalization_37" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="387">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="69" name="Clip_38" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="388">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="70" name="Multiply_9230" type="Const" version="opset1">
			<data element_type="f32" shape="192, 1, 1, 3, 3" offset="129028" size="6912" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="71" name="Multiply_8840" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="389">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="Constant_8845" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="135940" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="73" name="BatchNormalization_40" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="390">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="74" name="Clip_41" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="391">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="Multiply_9235" type="Const" version="opset1">
			<data element_type="f32" shape="32, 192, 1, 1" offset="136708" size="24576" />
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="76" name="Multiply_8847" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="Constant_8852" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="161284" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="78" name="BatchNormalization_43" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="393">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="79" name="Add_44" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="394">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="Multiply_9241" type="Const" version="opset1">
			<data element_type="f32" shape="192, 32, 1, 1" offset="161412" size="24576" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="81" name="Multiply_8854" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="Constant_8859" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="185988" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="83" name="BatchNormalization_46" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="396">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="84" name="Clip_47" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="397">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="85" name="Multiply_9247" type="Const" version="opset1">
			<data element_type="f32" shape="192, 1, 1, 3, 3" offset="186756" size="6912" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="86" name="Multiply_8861" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="398">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="Constant_8866" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="193668" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="88" name="BatchNormalization_49" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="399">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="89" name="Clip_50" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="400">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="Multiply_9252" type="Const" version="opset1">
			<data element_type="f32" shape="32, 192, 1, 1" offset="194436" size="24576" />
			<output>
				<port id="0" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="91" name="Multiply_8868" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="Constant_8873" type="Const" version="opset1">
			<data element_type="f32" shape="1, 32, 1, 1" offset="219012" size="128" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="93" name="BatchNormalization_52" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="402">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="94" name="Add_53" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="403">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="Multiply_9258" type="Const" version="opset1">
			<data element_type="f32" shape="192, 32, 1, 1" offset="219140" size="24576" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="96" name="Multiply_8875" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="Constant_8880" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="243716" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="98" name="BatchNormalization_55" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="405">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="Clip_56" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="406">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="100" name="Multiply_9264" type="Const" version="opset1">
			<data element_type="f32" shape="192, 1, 1, 3, 3" offset="244484" size="6912" />
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="101" name="Multiply_8882" type="GroupConvolution" version="opset1">
			<data strides="2, 2" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="407">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="Constant_8887" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="251396" size="768" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="103" name="BatchNormalization_58" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="408">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="104" name="Clip_59" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="409">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="Multiply_9269" type="Const" version="opset1">
			<data element_type="f32" shape="64, 192, 1, 1" offset="252164" size="49152" />
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="106" name="Multiply_8889" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="Constant_8894" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="301316" size="256" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="108" name="BatchNormalization_61" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="411">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="Multiply_9275" type="Const" version="opset1">
			<data element_type="f32" shape="384, 64, 1, 1" offset="301572" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="110" name="Multiply_8896" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="Constant_8901" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="399876" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="112" name="BatchNormalization_63" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="413">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="113" name="Clip_64" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="414">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="114" name="Multiply_9281" type="Const" version="opset1">
			<data element_type="f32" shape="384, 1, 1, 3, 3" offset="401412" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="115" name="Multiply_8903" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="415">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="Constant_8908" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="415236" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="117" name="BatchNormalization_66" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="416">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="118" name="Clip_67" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="417">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="Multiply_9286" type="Const" version="opset1">
			<data element_type="f32" shape="64, 384, 1, 1" offset="416772" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="120" name="Multiply_8910" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="Constant_8915" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="515076" size="256" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="122" name="BatchNormalization_69" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="419">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="123" name="Add_70" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="420">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="Multiply_9292" type="Const" version="opset1">
			<data element_type="f32" shape="384, 64, 1, 1" offset="515332" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="125" name="Multiply_8917" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="Constant_8922" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="613636" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="127" name="BatchNormalization_72" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="422">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="128" name="Clip_73" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="423">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="129" name="Multiply_9298" type="Const" version="opset1">
			<data element_type="f32" shape="384, 1, 1, 3, 3" offset="615172" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="130" name="Multiply_8924" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="424">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="Constant_8929" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="628996" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="132" name="BatchNormalization_75" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="425">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="133" name="Clip_76" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="426">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="Multiply_9303" type="Const" version="opset1">
			<data element_type="f32" shape="64, 384, 1, 1" offset="630532" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="135" name="Multiply_8931" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="Constant_8936" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="728836" size="256" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="137" name="BatchNormalization_78" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="428">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="Add_79" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="429">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="Multiply_9309" type="Const" version="opset1">
			<data element_type="f32" shape="384, 64, 1, 1" offset="729092" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="Multiply_8938" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="Constant_8943" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="827396" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="BatchNormalization_81" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="431">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="Clip_82" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="432">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="144" name="Multiply_9315" type="Const" version="opset1">
			<data element_type="f32" shape="384, 1, 1, 3, 3" offset="828932" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="Multiply_8945" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="433">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="Constant_8950" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="842756" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="BatchNormalization_84" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="434">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="Clip_85" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="435">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="Multiply_9320" type="Const" version="opset1">
			<data element_type="f32" shape="64, 384, 1, 1" offset="844292" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="150" name="Multiply_8952" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="Constant_8957" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="942596" size="256" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="152" name="BatchNormalization_87" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="437">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="153" name="Add_88" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="438">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="Multiply_9326" type="Const" version="opset1">
			<data element_type="f32" shape="384, 64, 1, 1" offset="942852" size="98304" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="155" name="Multiply_8959" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="Constant_8964" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="1041156" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="157" name="BatchNormalization_90" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="440">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="158" name="Clip_91" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="441">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="159" name="Multiply_9332" type="Const" version="opset1">
			<data element_type="f32" shape="384, 1, 1, 3, 3" offset="1042692" size="13824" />
			<output>
				<port id="0" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="160" name="Multiply_8966" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="442">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="Constant_8971" type="Const" version="opset1">
			<data element_type="f32" shape="1, 384, 1, 1" offset="1056516" size="1536" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="162" name="BatchNormalization_93" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="443">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="163" name="Clip_94" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="444">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="Multiply_9337" type="Const" version="opset1">
			<data element_type="f32" shape="96, 384, 1, 1" offset="1058052" size="147456" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="165" name="Multiply_8973" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="Constant_8978" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="1205508" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="167" name="BatchNormalization_96" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="446">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="Multiply_9343" type="Const" version="opset1">
			<data element_type="f32" shape="576, 96, 1, 1" offset="1205892" size="221184" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="169" name="Multiply_8980" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="Constant_8985" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="1427076" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="171" name="BatchNormalization_98" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="448">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="172" name="Clip_99" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="449">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="173" name="Multiply_9349" type="Const" version="opset1">
			<data element_type="f32" shape="576, 1, 1, 3, 3" offset="1429380" size="20736" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="174" name="Multiply_8987" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="450">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="Constant_8992" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="1450116" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="176" name="BatchNormalization_101" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="451">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="177" name="Clip_102" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="452">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="Multiply_9354" type="Const" version="opset1">
			<data element_type="f32" shape="96, 576, 1, 1" offset="1452420" size="221184" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="179" name="Multiply_8994" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="Constant_8999" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="1673604" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="181" name="BatchNormalization_104" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="454">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="182" name="Add_105" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="455">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="Multiply_9360" type="Const" version="opset1">
			<data element_type="f32" shape="576, 96, 1, 1" offset="1673988" size="221184" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="184" name="Multiply_9001" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="Constant_9006" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="1895172" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="186" name="BatchNormalization_107" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="457">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="187" name="Clip_108" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="458">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="188" name="Multiply_9366" type="Const" version="opset1">
			<data element_type="f32" shape="576, 1, 1, 3, 3" offset="1897476" size="20736" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="189" name="Multiply_9008" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="459">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="Constant_9013" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="1918212" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="191" name="BatchNormalization_110" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="460">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="192" name="Clip_111" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="461">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="Multiply_9371" type="Const" version="opset1">
			<data element_type="f32" shape="96, 576, 1, 1" offset="1920516" size="221184" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="194" name="Multiply_9015" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="Constant_9020" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="2141700" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="196" name="BatchNormalization_113" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="463">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="197" name="Add_114" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="464">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="Multiply_9377" type="Const" version="opset1">
			<data element_type="f32" shape="96, 1, 1, 3, 3" offset="2142084" size="3456" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="199" name="Multiply_9025" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="Constant_9030" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="2145540" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="201" name="BatchNormalization_154" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="504">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="202" name="Relu_155" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="505">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="203" name="bbox_head.reg_convs.0.3.weight" type="Const" version="opset1">
			<data element_type="f32" shape="16, 96, 1, 1" offset="2145924" size="6144" />
			<output>
				<port id="0" precision="FP32" names="bbox_head.reg_convs.0.3.weight">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="204" name="Conv_156/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="Reshape_1480" type="Const" version="opset1">
			<data element_type="f32" shape="1, 16, 1, 1" offset="2152068" size="64" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="206" name="Conv_156" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="506">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="207" name="Constant_1760" type="Const" version="opset1">
			<data element_type="i64" shape="4" offset="2152132" size="32" />
			<output>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="Transpose_196" type="Transpose" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="556">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="Shape_197" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64" names="557">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="Constant_4924" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="Constant_1763" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64" />
			</output>
		</layer>
		<layer id="212" name="Gather_199" type="Gather" version="opset8">
			<data batch_dims="0" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64" />
			</input>
			<output>
				<port id="3" precision="I64" names="561">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="581" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152172" size="8" />
			<output>
				<port id="0" precision="I64" names="581">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="214" name="Concat_201" type="Concat" version="opset1">
			<data axis="0" />
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64" names="563">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="215" name="Reshape_202" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="564">
					<dim>1</dim>
					<dim>4096</dim>
				</port>
			</output>
		</layer>
		<layer id="216" name="Multiply_9382" type="Const" version="opset1">
			<data element_type="f32" shape="576, 96, 1, 1" offset="2152180" size="221184" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="217" name="Multiply_9032" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="Constant_9037" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="2373364" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="219" name="BatchNormalization_116" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="466">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="Clip_117" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="467">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="221" name="Multiply_9388" type="Const" version="opset1">
			<data element_type="f32" shape="576, 1, 1, 3, 3" offset="2375668" size="20736" />
			<output>
				<port id="0" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="Multiply_9039" type="GroupConvolution" version="opset1">
			<data strides="2, 2" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="468">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="223" name="Constant_9044" type="Const" version="opset1">
			<data element_type="f32" shape="1, 576, 1, 1" offset="2396404" size="2304" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="224" name="BatchNormalization_119" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="469">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="Clip_120" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="470">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="226" name="Multiply_9393" type="Const" version="opset1">
			<data element_type="f32" shape="160, 576, 1, 1" offset="2398708" size="368640" />
			<output>
				<port id="0" precision="FP32">
					<dim>160</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="227" name="Multiply_9046" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="Constant_9051" type="Const" version="opset1">
			<data element_type="f32" shape="1, 160, 1, 1" offset="2767348" size="640" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="229" name="BatchNormalization_122" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="472">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="Multiply_9399" type="Const" version="opset1">
			<data element_type="f32" shape="960, 160, 1, 1" offset="2767988" size="614400" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="231" name="Multiply_9053" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="Constant_9058" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="3382388" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="233" name="BatchNormalization_124" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="474">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="Clip_125" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="475">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="235" name="Multiply_9405" type="Const" version="opset1">
			<data element_type="f32" shape="960, 1, 1, 3, 3" offset="3386228" size="34560" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="Multiply_9060" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="476">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="237" name="Constant_9065" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="3420788" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="238" name="BatchNormalization_127" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="477">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="Clip_128" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="478">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="240" name="Multiply_9410" type="Const" version="opset1">
			<data element_type="f32" shape="160, 960, 1, 1" offset="3424628" size="614400" />
			<output>
				<port id="0" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="241" name="Multiply_9067" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="Constant_9072" type="Const" version="opset1">
			<data element_type="f32" shape="1, 160, 1, 1" offset="4039028" size="640" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="243" name="BatchNormalization_130" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="480">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="Add_131" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="481">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="245" name="Multiply_9416" type="Const" version="opset1">
			<data element_type="f32" shape="960, 160, 1, 1" offset="4039668" size="614400" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="246" name="Multiply_9074" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="Constant_9079" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="4654068" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="248" name="BatchNormalization_133" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="483">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="Clip_134" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="484">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="250" name="Multiply_9422" type="Const" version="opset1">
			<data element_type="f32" shape="960, 1, 1, 3, 3" offset="4657908" size="34560" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="251" name="Multiply_9081" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="485">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="252" name="Constant_9086" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="4692468" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="253" name="BatchNormalization_136" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="486">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="Clip_137" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="487">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="255" name="Multiply_9427" type="Const" version="opset1">
			<data element_type="f32" shape="160, 960, 1, 1" offset="4696308" size="614400" />
			<output>
				<port id="0" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="256" name="Multiply_9088" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="Constant_9093" type="Const" version="opset1">
			<data element_type="f32" shape="1, 160, 1, 1" offset="5310708" size="640" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="258" name="BatchNormalization_139" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="489">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="Add_140" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="490">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="260" name="Multiply_9433" type="Const" version="opset1">
			<data element_type="f32" shape="960, 160, 1, 1" offset="5311348" size="614400" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="261" name="Multiply_9095" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="Constant_9100" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="5925748" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="263" name="BatchNormalization_142" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="492">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="264" name="Clip_143" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="493">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="265" name="Multiply_9439" type="Const" version="opset1">
			<data element_type="f32" shape="960, 1, 1, 3, 3" offset="5929588" size="34560" />
			<output>
				<port id="0" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="266" name="Multiply_9102" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="494">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="267" name="Constant_9107" type="Const" version="opset1">
			<data element_type="f32" shape="1, 960, 1, 1" offset="5964148" size="3840" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="268" name="BatchNormalization_145" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="495">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="Clip_146" type="Clamp" version="opset1">
			<data min="0" max="6" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="496">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="270" name="Multiply_9444" type="Const" version="opset1">
			<data element_type="f32" shape="320, 960, 1, 1" offset="5967988" size="1228800" />
			<output>
				<port id="0" precision="FP32">
					<dim>320</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="271" name="Multiply_9109" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="Constant_9114" type="Const" version="opset1">
			<data element_type="f32" shape="1, 320, 1, 1" offset="7196788" size="1280" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="273" name="BatchNormalization_148" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="498">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="Multiply_9450" type="Const" version="opset1">
			<data element_type="f32" shape="320, 1, 1, 3, 3" offset="7198068" size="11520" />
			<output>
				<port id="0" precision="FP32">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="275" name="Multiply_9119" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="Constant_9124" type="Const" version="opset1">
			<data element_type="f32" shape="1, 320, 1, 1" offset="7209588" size="1280" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="BatchNormalization_162" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="512">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="Relu_163" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="513">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="bbox_head.reg_convs.1.3.weight" type="Const" version="opset1">
			<data element_type="f32" shape="20, 320, 1, 1" offset="7210868" size="25600" />
			<output>
				<port id="0" precision="FP32" names="bbox_head.reg_convs.1.3.weight">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="Conv_164/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="Reshape_1640" type="Const" version="opset1">
			<data element_type="f32" shape="1, 20, 1, 1" offset="7236468" size="80" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="Conv_164" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="514">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="Constant_1771" type="Const" version="opset1">
			<data element_type="i64" shape="4" offset="2152132" size="32" />
			<output>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="Transpose_203" type="Transpose" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="565">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="Shape_204" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64" names="566">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="Constant_4930" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="Constant_1774" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64" />
			</output>
		</layer>
		<layer id="288" name="Gather_206" type="Gather" version="opset8">
			<data batch_dims="0" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64" />
			</input>
			<output>
				<port id="3" precision="I64" names="570">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="289" name="582" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152172" size="8" />
			<output>
				<port id="0" precision="I64" names="582">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="290" name="Concat_208" type="Concat" version="opset1">
			<data axis="0" />
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64" names="572">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="291" name="Reshape_209" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>20</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="573">
					<dim>1</dim>
					<dim>1280</dim>
				</port>
			</output>
		</layer>
		<layer id="292" name="Concat_210" type="Concat" version="opset1">
			<data axis="1" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4096</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="574">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
			</output>
		</layer>
		<layer id="293" name="Multiply_9455" type="Const" version="opset1">
			<data element_type="f32" shape="96, 1, 1, 3, 3" offset="7236548" size="3456" />
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="294" name="Multiply_9129" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="295" name="Constant_9134" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="7240004" size="384" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="BatchNormalization_150" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="500">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="297" name="Relu_151" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="501">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="298" name="bbox_head.cls_convs.0.3.weight" type="Const" version="opset1">
			<data element_type="f32" shape="16, 96, 1, 1" offset="7240388" size="6144" />
			<output>
				<port id="0" precision="FP32" names="bbox_head.cls_convs.0.3.weight">
					<dim>16</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="299" name="Conv_152/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="300" name="Reshape_1400" type="Const" version="opset1">
			<data element_type="f32" shape="1, 16, 1, 1" offset="7246532" size="64" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="Conv_152" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="502">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="Constant_1673" type="Const" version="opset1">
			<data element_type="i64" shape="4" offset="2152132" size="32" />
			<output>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="Transpose_168" type="Transpose" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="518">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="304" name="Shape_169" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64" names="519">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="305" name="Constant_4936" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="306" name="Constant_1676" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64" />
			</output>
		</layer>
		<layer id="307" name="Gather_171" type="Gather" version="opset8">
			<data batch_dims="0" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64" />
			</input>
			<output>
				<port id="3" precision="I64" names="523">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="308" name="576" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152172" size="8" />
			<output>
				<port id="0" precision="I64" names="576">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="309" name="Concat_173" type="Concat" version="opset1">
			<data axis="0" />
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64" names="525">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="310" name="Reshape_174" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="526">
					<dim>1</dim>
					<dim>4096</dim>
				</port>
			</output>
		</layer>
		<layer id="311" name="Multiply_9460" type="Const" version="opset1">
			<data element_type="f32" shape="320, 1, 1, 3, 3" offset="7246596" size="11520" />
			<output>
				<port id="0" precision="FP32">
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="312" name="Multiply_9139" type="GroupConvolution" version="opset1">
			<data strides="1, 1" pads_begin="1, 1" pads_end="1, 1" dilations="1, 1" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="313" name="Constant_9144" type="Const" version="opset1">
			<data element_type="f32" shape="1, 320, 1, 1" offset="7258116" size="1280" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="314" name="BatchNormalization_158" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="508">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="315" name="Relu_159" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="509">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="316" name="bbox_head.cls_convs.1.3.weight" type="Const" version="opset1">
			<data element_type="f32" shape="20, 320, 1, 1" offset="7259396" size="25600" />
			<output>
				<port id="0" precision="FP32" names="bbox_head.cls_convs.1.3.weight">
					<dim>20</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="317" name="Conv_160/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="318" name="Reshape_1560" type="Const" version="opset1">
			<data element_type="f32" shape="1, 20, 1, 1" offset="7284996" size="80" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="319" name="Conv_160" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="510">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="320" name="Constant_1684" type="Const" version="opset1">
			<data element_type="i64" shape="4" offset="2152132" size="32" />
			<output>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="321" name="Transpose_175" type="Transpose" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="527">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="322" name="Shape_176" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>20</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64" names="528">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="323" name="Constant_4942" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="324" name="Constant_1687" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64" />
			</output>
		</layer>
		<layer id="325" name="Gather_178" type="Gather" version="opset8">
			<data batch_dims="0" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64" />
			</input>
			<output>
				<port id="3" precision="I64" names="532">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="326" name="577" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152172" size="8" />
			<output>
				<port id="0" precision="I64" names="577">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="327" name="Concat_180" type="Concat" version="opset1">
			<data axis="0" />
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64" names="534">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="328" name="Reshape_181" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>20</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="535">
					<dim>1</dim>
					<dim>1280</dim>
				</port>
			</output>
		</layer>
		<layer id="329" name="Concat_182" type="Concat" version="opset1">
			<data axis="1" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4096</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="536">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
			</output>
		</layer>
		<layer id="330" name="Concat_187" type="Const" version="opset1">
			<data element_type="i64" shape="3" offset="7285076" size="24" />
			<output>
				<port id="0" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="331" name="Reshape_188" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="546">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="332" name="ShapeOf_1705" type="ShapeOf" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="333" name="Constant_1708" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="334" name="Constant_1707" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285100" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="335" name="Constant_1709" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285108" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="336" name="StridedSlice_1710" type="StridedSlice" version="opset1">
			<data begin_mask="0" end_mask="0" new_axis_mask="" shrink_axis_mask="" ellipsis_mask="" />
			<input>
				<port id="0" precision="I64">
					<dim>3</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="337" name="Constant_1711" type="Const" version="opset1">
			<data element_type="i64" shape="" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64" />
			</output>
		</layer>
		<layer id="338" name="ReduceProd_1712" type="ReduceProd" version="opset1">
			<data keep_dims="true" />
			<input>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
				<port id="1" precision="I64" />
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="339" name="Constant_1713" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152172" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="340" name="Concat_1714" type="Concat" version="opset1">
			<data axis="0" />
			<input>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="341" name="Reshape_1715" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="342" name="Softmax_1746" type="SoftMax" version="opset8">
			<data axis="1" />
			<input>
				<port id="0" precision="FP32">
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="343" name="Softmax_189" type="Reshape" version="opset1">
			<data special_zero="false" />
			<input>
				<port id="0" precision="FP32">
					<dim>1344</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="547">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="344" name="Concat_194" type="Const" version="opset1">
			<data element_type="i64" shape="2" offset="7285116" size="16" />
			<output>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="345" name="Reshape_195" type="Reshape" version="opset1">
			<data special_zero="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1344</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="555">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
			</output>
		</layer>
		<layer id="346" name="ShapeOf_1644" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
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
		<layer id="347" name="Constant_1647" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285100" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="348" name="Constant_1646" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285132" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="349" name="Constant_1648" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285108" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="350" name="StridedSlice_1649" type="StridedSlice" version="opset1">
			<data begin_mask="0" end_mask="0" new_axis_mask="" shrink_axis_mask="" ellipsis_mask="" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="351" name="ShapeOf_1645" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>256</dim>
					<dim>256</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="352" name="Constant_1651" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285100" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="353" name="Constant_1650" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285132" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="354" name="Constant_1652" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285108" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="355" name="StridedSlice_1653" type="StridedSlice" version="opset1">
			<data begin_mask="0" end_mask="0" new_axis_mask="" shrink_axis_mask="" ellipsis_mask="" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="356" name="PriorBoxClustered_1655" type="PriorBoxClustered" version="opset1">
			<data step="16" step_w="0" step_h="0" width="7.16418, 13.0066, 15.3153, 23.5724" height="12.2863, 29.1984, 53.5002, 85.3917" clip="false" offset="0.5" variance="0.1, 0.1, 0.2, 0.2" />
			<input>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>4096</dim>
				</port>
			</output>
		</layer>
		<layer id="357" name="Constant_1654" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="358" name="PriorBoxClustered_165" type="Unsqueeze" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>4096</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="515">
					<dim>1</dim>
					<dim>2</dim>
					<dim>4096</dim>
				</port>
			</output>
		</layer>
		<layer id="359" name="ShapeOf_1658" type="ShapeOf" version="opset3">
			<data output_type="i64" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="360" name="Constant_1661" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285100" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="361" name="Constant_1660" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285132" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="362" name="Constant_1662" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285108" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="363" name="StridedSlice_1663" type="StridedSlice" version="opset1">
			<data begin_mask="0" end_mask="0" new_axis_mask="" shrink_axis_mask="" ellipsis_mask="" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="364" name="Constant_1665" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285100" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="365" name="Constant_1664" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285132" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="366" name="Constant_1666" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="7285108" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="367" name="StridedSlice_1667" type="StridedSlice" version="opset1">
			<data begin_mask="0" end_mask="0" new_axis_mask="" shrink_axis_mask="" ellipsis_mask="" />
			<input>
				<port id="0" precision="I64">
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
				</port>
				<port id="3" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="368" name="PriorBoxClustered_1669" type="PriorBoxClustered" version="opset1">
			<data step="32" step_w="0" step_h="0" width="51.8645, 32.5943, 49.53, 91.6219, 150.333" height="42.8697, 126.526, 190.253, 105.12, 146.429" clip="false" offset="0.5" variance="0.1, 0.1, 0.2, 0.2" />
			<input>
				<port id="0" precision="I64">
					<dim>2</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>1280</dim>
				</port>
			</output>
		</layer>
		<layer id="369" name="Constant_1668" type="Const" version="opset1">
			<data element_type="i64" shape="1" offset="2152164" size="8" />
			<output>
				<port id="0" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="370" name="PriorBoxClustered_166" type="Unsqueeze" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>1280</dim>
				</port>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="516">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1280</dim>
				</port>
			</output>
		</layer>
		<layer id="371" name="Concat_167" type="Concat" version="opset1">
			<data axis="2" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>4096</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1280</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="517">
					<dim>1</dim>
					<dim>2</dim>
					<dim>5376</dim>
				</port>
			</output>
		</layer>
		<layer id="372" name="detection_out" type="DetectionOutput" version="opset8">
			<data background_label_id="3" top_k="200" variance_encoded_in_target="false" keep_top_k="200" code_type="caffe.PriorBoxParameter.CENTER_SIZE" share_location="true" nms_threshold="0.44999998807907104" confidence_threshold="0.019999999552965164" clip_after_nms="false" clip_before_nms="false" decrease_label_id="false" normalized="true" input_height="1" input_width="1" objectness_score="0" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>5376</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>5376</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32" names="detection_out">
					<dim>1</dim>
					<dim>1</dim>
					<dim>200</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="373" name="detection_out/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>200</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="351" to-port="0" />
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="0" to-layer="4" to-port="1" />
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0" />
		<edge from-layer="5" from-port="0" to-layer="6" to-port="1" />
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0" />
		<edge from-layer="7" from-port="1" to-layer="9" to-port="0" />
		<edge from-layer="8" from-port="0" to-layer="9" to-port="1" />
		<edge from-layer="9" from-port="2" to-layer="11" to-port="0" />
		<edge from-layer="10" from-port="0" to-layer="11" to-port="1" />
		<edge from-layer="11" from-port="2" to-layer="12" to-port="0" />
		<edge from-layer="12" from-port="1" to-layer="14" to-port="0" />
		<edge from-layer="13" from-port="0" to-layer="14" to-port="1" />
		<edge from-layer="14" from-port="2" to-layer="16" to-port="0" />
		<edge from-layer="15" from-port="0" to-layer="16" to-port="1" />
		<edge from-layer="16" from-port="2" to-layer="17" to-port="0" />
		<edge from-layer="17" from-port="1" to-layer="19" to-port="0" />
		<edge from-layer="18" from-port="0" to-layer="19" to-port="1" />
		<edge from-layer="19" from-port="2" to-layer="21" to-port="0" />
		<edge from-layer="20" from-port="0" to-layer="21" to-port="1" />
		<edge from-layer="21" from-port="2" to-layer="23" to-port="0" />
		<edge from-layer="22" from-port="0" to-layer="23" to-port="1" />
		<edge from-layer="23" from-port="2" to-layer="25" to-port="0" />
		<edge from-layer="24" from-port="0" to-layer="25" to-port="1" />
		<edge from-layer="25" from-port="2" to-layer="26" to-port="0" />
		<edge from-layer="26" from-port="1" to-layer="28" to-port="0" />
		<edge from-layer="27" from-port="0" to-layer="28" to-port="1" />
		<edge from-layer="28" from-port="2" to-layer="30" to-port="0" />
		<edge from-layer="29" from-port="0" to-layer="30" to-port="1" />
		<edge from-layer="30" from-port="2" to-layer="31" to-port="0" />
		<edge from-layer="31" from-port="1" to-layer="33" to-port="0" />
		<edge from-layer="32" from-port="0" to-layer="33" to-port="1" />
		<edge from-layer="33" from-port="2" to-layer="35" to-port="0" />
		<edge from-layer="34" from-port="0" to-layer="35" to-port="1" />
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0" />
		<edge from-layer="35" from-port="2" to-layer="50" to-port="1" />
		<edge from-layer="36" from-port="0" to-layer="37" to-port="1" />
		<edge from-layer="37" from-port="2" to-layer="39" to-port="0" />
		<edge from-layer="38" from-port="0" to-layer="39" to-port="1" />
		<edge from-layer="39" from-port="2" to-layer="40" to-port="0" />
		<edge from-layer="40" from-port="1" to-layer="42" to-port="0" />
		<edge from-layer="41" from-port="0" to-layer="42" to-port="1" />
		<edge from-layer="42" from-port="2" to-layer="44" to-port="0" />
		<edge from-layer="43" from-port="0" to-layer="44" to-port="1" />
		<edge from-layer="44" from-port="2" to-layer="45" to-port="0" />
		<edge from-layer="45" from-port="1" to-layer="47" to-port="0" />
		<edge from-layer="46" from-port="0" to-layer="47" to-port="1" />
		<edge from-layer="47" from-port="2" to-layer="49" to-port="0" />
		<edge from-layer="48" from-port="0" to-layer="49" to-port="1" />
		<edge from-layer="49" from-port="2" to-layer="50" to-port="0" />
		<edge from-layer="50" from-port="2" to-layer="52" to-port="0" />
		<edge from-layer="51" from-port="0" to-layer="52" to-port="1" />
		<edge from-layer="52" from-port="2" to-layer="54" to-port="0" />
		<edge from-layer="53" from-port="0" to-layer="54" to-port="1" />
		<edge from-layer="54" from-port="2" to-layer="55" to-port="0" />
		<edge from-layer="55" from-port="1" to-layer="57" to-port="0" />
		<edge from-layer="56" from-port="0" to-layer="57" to-port="1" />
		<edge from-layer="57" from-port="2" to-layer="59" to-port="0" />
		<edge from-layer="58" from-port="0" to-layer="59" to-port="1" />
		<edge from-layer="59" from-port="2" to-layer="60" to-port="0" />
		<edge from-layer="60" from-port="1" to-layer="62" to-port="0" />
		<edge from-layer="61" from-port="0" to-layer="62" to-port="1" />
		<edge from-layer="62" from-port="2" to-layer="64" to-port="0" />
		<edge from-layer="63" from-port="0" to-layer="64" to-port="1" />
		<edge from-layer="64" from-port="2" to-layer="79" to-port="1" />
		<edge from-layer="64" from-port="2" to-layer="66" to-port="0" />
		<edge from-layer="65" from-port="0" to-layer="66" to-port="1" />
		<edge from-layer="66" from-port="2" to-layer="68" to-port="0" />
		<edge from-layer="67" from-port="0" to-layer="68" to-port="1" />
		<edge from-layer="68" from-port="2" to-layer="69" to-port="0" />
		<edge from-layer="69" from-port="1" to-layer="71" to-port="0" />
		<edge from-layer="70" from-port="0" to-layer="71" to-port="1" />
		<edge from-layer="71" from-port="2" to-layer="73" to-port="0" />
		<edge from-layer="72" from-port="0" to-layer="73" to-port="1" />
		<edge from-layer="73" from-port="2" to-layer="74" to-port="0" />
		<edge from-layer="74" from-port="1" to-layer="76" to-port="0" />
		<edge from-layer="75" from-port="0" to-layer="76" to-port="1" />
		<edge from-layer="76" from-port="2" to-layer="78" to-port="0" />
		<edge from-layer="77" from-port="0" to-layer="78" to-port="1" />
		<edge from-layer="78" from-port="2" to-layer="79" to-port="0" />
		<edge from-layer="79" from-port="2" to-layer="81" to-port="0" />
		<edge from-layer="79" from-port="2" to-layer="94" to-port="1" />
		<edge from-layer="80" from-port="0" to-layer="81" to-port="1" />
		<edge from-layer="81" from-port="2" to-layer="83" to-port="0" />
		<edge from-layer="82" from-port="0" to-layer="83" to-port="1" />
		<edge from-layer="83" from-port="2" to-layer="84" to-port="0" />
		<edge from-layer="84" from-port="1" to-layer="86" to-port="0" />
		<edge from-layer="85" from-port="0" to-layer="86" to-port="1" />
		<edge from-layer="86" from-port="2" to-layer="88" to-port="0" />
		<edge from-layer="87" from-port="0" to-layer="88" to-port="1" />
		<edge from-layer="88" from-port="2" to-layer="89" to-port="0" />
		<edge from-layer="89" from-port="1" to-layer="91" to-port="0" />
		<edge from-layer="90" from-port="0" to-layer="91" to-port="1" />
		<edge from-layer="91" from-port="2" to-layer="93" to-port="0" />
		<edge from-layer="92" from-port="0" to-layer="93" to-port="1" />
		<edge from-layer="93" from-port="2" to-layer="94" to-port="0" />
		<edge from-layer="94" from-port="2" to-layer="96" to-port="0" />
		<edge from-layer="95" from-port="0" to-layer="96" to-port="1" />
		<edge from-layer="96" from-port="2" to-layer="98" to-port="0" />
		<edge from-layer="97" from-port="0" to-layer="98" to-port="1" />
		<edge from-layer="98" from-port="2" to-layer="99" to-port="0" />
		<edge from-layer="99" from-port="1" to-layer="101" to-port="0" />
		<edge from-layer="100" from-port="0" to-layer="101" to-port="1" />
		<edge from-layer="101" from-port="2" to-layer="103" to-port="0" />
		<edge from-layer="102" from-port="0" to-layer="103" to-port="1" />
		<edge from-layer="103" from-port="2" to-layer="104" to-port="0" />
		<edge from-layer="104" from-port="1" to-layer="106" to-port="0" />
		<edge from-layer="105" from-port="0" to-layer="106" to-port="1" />
		<edge from-layer="106" from-port="2" to-layer="108" to-port="0" />
		<edge from-layer="107" from-port="0" to-layer="108" to-port="1" />
		<edge from-layer="108" from-port="2" to-layer="110" to-port="0" />
		<edge from-layer="108" from-port="2" to-layer="123" to-port="1" />
		<edge from-layer="109" from-port="0" to-layer="110" to-port="1" />
		<edge from-layer="110" from-port="2" to-layer="112" to-port="0" />
		<edge from-layer="111" from-port="0" to-layer="112" to-port="1" />
		<edge from-layer="112" from-port="2" to-layer="113" to-port="0" />
		<edge from-layer="113" from-port="1" to-layer="115" to-port="0" />
		<edge from-layer="114" from-port="0" to-layer="115" to-port="1" />
		<edge from-layer="115" from-port="2" to-layer="117" to-port="0" />
		<edge from-layer="116" from-port="0" to-layer="117" to-port="1" />
		<edge from-layer="117" from-port="2" to-layer="118" to-port="0" />
		<edge from-layer="118" from-port="1" to-layer="120" to-port="0" />
		<edge from-layer="119" from-port="0" to-layer="120" to-port="1" />
		<edge from-layer="120" from-port="2" to-layer="122" to-port="0" />
		<edge from-layer="121" from-port="0" to-layer="122" to-port="1" />
		<edge from-layer="122" from-port="2" to-layer="123" to-port="0" />
		<edge from-layer="123" from-port="2" to-layer="125" to-port="0" />
		<edge from-layer="123" from-port="2" to-layer="138" to-port="1" />
		<edge from-layer="124" from-port="0" to-layer="125" to-port="1" />
		<edge from-layer="125" from-port="2" to-layer="127" to-port="0" />
		<edge from-layer="126" from-port="0" to-layer="127" to-port="1" />
		<edge from-layer="127" from-port="2" to-layer="128" to-port="0" />
		<edge from-layer="128" from-port="1" to-layer="130" to-port="0" />
		<edge from-layer="129" from-port="0" to-layer="130" to-port="1" />
		<edge from-layer="130" from-port="2" to-layer="132" to-port="0" />
		<edge from-layer="131" from-port="0" to-layer="132" to-port="1" />
		<edge from-layer="132" from-port="2" to-layer="133" to-port="0" />
		<edge from-layer="133" from-port="1" to-layer="135" to-port="0" />
		<edge from-layer="134" from-port="0" to-layer="135" to-port="1" />
		<edge from-layer="135" from-port="2" to-layer="137" to-port="0" />
		<edge from-layer="136" from-port="0" to-layer="137" to-port="1" />
		<edge from-layer="137" from-port="2" to-layer="138" to-port="0" />
		<edge from-layer="138" from-port="2" to-layer="153" to-port="1" />
		<edge from-layer="138" from-port="2" to-layer="140" to-port="0" />
		<edge from-layer="139" from-port="0" to-layer="140" to-port="1" />
		<edge from-layer="140" from-port="2" to-layer="142" to-port="0" />
		<edge from-layer="141" from-port="0" to-layer="142" to-port="1" />
		<edge from-layer="142" from-port="2" to-layer="143" to-port="0" />
		<edge from-layer="143" from-port="1" to-layer="145" to-port="0" />
		<edge from-layer="144" from-port="0" to-layer="145" to-port="1" />
		<edge from-layer="145" from-port="2" to-layer="147" to-port="0" />
		<edge from-layer="146" from-port="0" to-layer="147" to-port="1" />
		<edge from-layer="147" from-port="2" to-layer="148" to-port="0" />
		<edge from-layer="148" from-port="1" to-layer="150" to-port="0" />
		<edge from-layer="149" from-port="0" to-layer="150" to-port="1" />
		<edge from-layer="150" from-port="2" to-layer="152" to-port="0" />
		<edge from-layer="151" from-port="0" to-layer="152" to-port="1" />
		<edge from-layer="152" from-port="2" to-layer="153" to-port="0" />
		<edge from-layer="153" from-port="2" to-layer="155" to-port="0" />
		<edge from-layer="154" from-port="0" to-layer="155" to-port="1" />
		<edge from-layer="155" from-port="2" to-layer="157" to-port="0" />
		<edge from-layer="156" from-port="0" to-layer="157" to-port="1" />
		<edge from-layer="157" from-port="2" to-layer="158" to-port="0" />
		<edge from-layer="158" from-port="1" to-layer="160" to-port="0" />
		<edge from-layer="159" from-port="0" to-layer="160" to-port="1" />
		<edge from-layer="160" from-port="2" to-layer="162" to-port="0" />
		<edge from-layer="161" from-port="0" to-layer="162" to-port="1" />
		<edge from-layer="162" from-port="2" to-layer="163" to-port="0" />
		<edge from-layer="163" from-port="1" to-layer="165" to-port="0" />
		<edge from-layer="164" from-port="0" to-layer="165" to-port="1" />
		<edge from-layer="165" from-port="2" to-layer="167" to-port="0" />
		<edge from-layer="166" from-port="0" to-layer="167" to-port="1" />
		<edge from-layer="167" from-port="2" to-layer="169" to-port="0" />
		<edge from-layer="167" from-port="2" to-layer="182" to-port="1" />
		<edge from-layer="168" from-port="0" to-layer="169" to-port="1" />
		<edge from-layer="169" from-port="2" to-layer="171" to-port="0" />
		<edge from-layer="170" from-port="0" to-layer="171" to-port="1" />
		<edge from-layer="171" from-port="2" to-layer="172" to-port="0" />
		<edge from-layer="172" from-port="1" to-layer="174" to-port="0" />
		<edge from-layer="173" from-port="0" to-layer="174" to-port="1" />
		<edge from-layer="174" from-port="2" to-layer="176" to-port="0" />
		<edge from-layer="175" from-port="0" to-layer="176" to-port="1" />
		<edge from-layer="176" from-port="2" to-layer="177" to-port="0" />
		<edge from-layer="177" from-port="1" to-layer="179" to-port="0" />
		<edge from-layer="178" from-port="0" to-layer="179" to-port="1" />
		<edge from-layer="179" from-port="2" to-layer="181" to-port="0" />
		<edge from-layer="180" from-port="0" to-layer="181" to-port="1" />
		<edge from-layer="181" from-port="2" to-layer="182" to-port="0" />
		<edge from-layer="182" from-port="2" to-layer="197" to-port="1" />
		<edge from-layer="182" from-port="2" to-layer="184" to-port="0" />
		<edge from-layer="183" from-port="0" to-layer="184" to-port="1" />
		<edge from-layer="184" from-port="2" to-layer="186" to-port="0" />
		<edge from-layer="185" from-port="0" to-layer="186" to-port="1" />
		<edge from-layer="186" from-port="2" to-layer="187" to-port="0" />
		<edge from-layer="187" from-port="1" to-layer="189" to-port="0" />
		<edge from-layer="188" from-port="0" to-layer="189" to-port="1" />
		<edge from-layer="189" from-port="2" to-layer="191" to-port="0" />
		<edge from-layer="190" from-port="0" to-layer="191" to-port="1" />
		<edge from-layer="191" from-port="2" to-layer="192" to-port="0" />
		<edge from-layer="192" from-port="1" to-layer="194" to-port="0" />
		<edge from-layer="193" from-port="0" to-layer="194" to-port="1" />
		<edge from-layer="194" from-port="2" to-layer="196" to-port="0" />
		<edge from-layer="195" from-port="0" to-layer="196" to-port="1" />
		<edge from-layer="196" from-port="2" to-layer="197" to-port="0" />
		<edge from-layer="197" from-port="2" to-layer="294" to-port="0" />
		<edge from-layer="197" from-port="2" to-layer="346" to-port="0" />
		<edge from-layer="197" from-port="2" to-layer="217" to-port="0" />
		<edge from-layer="197" from-port="2" to-layer="199" to-port="0" />
		<edge from-layer="198" from-port="0" to-layer="199" to-port="1" />
		<edge from-layer="199" from-port="2" to-layer="201" to-port="0" />
		<edge from-layer="200" from-port="0" to-layer="201" to-port="1" />
		<edge from-layer="201" from-port="2" to-layer="202" to-port="0" />
		<edge from-layer="202" from-port="1" to-layer="204" to-port="0" />
		<edge from-layer="203" from-port="0" to-layer="204" to-port="1" />
		<edge from-layer="204" from-port="2" to-layer="206" to-port="0" />
		<edge from-layer="205" from-port="0" to-layer="206" to-port="1" />
		<edge from-layer="206" from-port="2" to-layer="208" to-port="0" />
		<edge from-layer="206" from-port="2" to-layer="209" to-port="0" />
		<edge from-layer="207" from-port="0" to-layer="208" to-port="1" />
		<edge from-layer="208" from-port="2" to-layer="215" to-port="0" />
		<edge from-layer="209" from-port="1" to-layer="212" to-port="0" />
		<edge from-layer="210" from-port="0" to-layer="212" to-port="1" />
		<edge from-layer="211" from-port="0" to-layer="212" to-port="2" />
		<edge from-layer="212" from-port="3" to-layer="214" to-port="0" />
		<edge from-layer="213" from-port="0" to-layer="214" to-port="1" />
		<edge from-layer="214" from-port="2" to-layer="215" to-port="1" />
		<edge from-layer="215" from-port="2" to-layer="292" to-port="0" />
		<edge from-layer="216" from-port="0" to-layer="217" to-port="1" />
		<edge from-layer="217" from-port="2" to-layer="219" to-port="0" />
		<edge from-layer="218" from-port="0" to-layer="219" to-port="1" />
		<edge from-layer="219" from-port="2" to-layer="220" to-port="0" />
		<edge from-layer="220" from-port="1" to-layer="222" to-port="0" />
		<edge from-layer="221" from-port="0" to-layer="222" to-port="1" />
		<edge from-layer="222" from-port="2" to-layer="224" to-port="0" />
		<edge from-layer="223" from-port="0" to-layer="224" to-port="1" />
		<edge from-layer="224" from-port="2" to-layer="225" to-port="0" />
		<edge from-layer="225" from-port="1" to-layer="227" to-port="0" />
		<edge from-layer="226" from-port="0" to-layer="227" to-port="1" />
		<edge from-layer="227" from-port="2" to-layer="229" to-port="0" />
		<edge from-layer="228" from-port="0" to-layer="229" to-port="1" />
		<edge from-layer="229" from-port="2" to-layer="231" to-port="0" />
		<edge from-layer="229" from-port="2" to-layer="244" to-port="1" />
		<edge from-layer="230" from-port="0" to-layer="231" to-port="1" />
		<edge from-layer="231" from-port="2" to-layer="233" to-port="0" />
		<edge from-layer="232" from-port="0" to-layer="233" to-port="1" />
		<edge from-layer="233" from-port="2" to-layer="234" to-port="0" />
		<edge from-layer="234" from-port="1" to-layer="236" to-port="0" />
		<edge from-layer="235" from-port="0" to-layer="236" to-port="1" />
		<edge from-layer="236" from-port="2" to-layer="238" to-port="0" />
		<edge from-layer="237" from-port="0" to-layer="238" to-port="1" />
		<edge from-layer="238" from-port="2" to-layer="239" to-port="0" />
		<edge from-layer="239" from-port="1" to-layer="241" to-port="0" />
		<edge from-layer="240" from-port="0" to-layer="241" to-port="1" />
		<edge from-layer="241" from-port="2" to-layer="243" to-port="0" />
		<edge from-layer="242" from-port="0" to-layer="243" to-port="1" />
		<edge from-layer="243" from-port="2" to-layer="244" to-port="0" />
		<edge from-layer="244" from-port="2" to-layer="246" to-port="0" />
		<edge from-layer="244" from-port="2" to-layer="259" to-port="1" />
		<edge from-layer="245" from-port="0" to-layer="246" to-port="1" />
		<edge from-layer="246" from-port="2" to-layer="248" to-port="0" />
		<edge from-layer="247" from-port="0" to-layer="248" to-port="1" />
		<edge from-layer="248" from-port="2" to-layer="249" to-port="0" />
		<edge from-layer="249" from-port="1" to-layer="251" to-port="0" />
		<edge from-layer="250" from-port="0" to-layer="251" to-port="1" />
		<edge from-layer="251" from-port="2" to-layer="253" to-port="0" />
		<edge from-layer="252" from-port="0" to-layer="253" to-port="1" />
		<edge from-layer="253" from-port="2" to-layer="254" to-port="0" />
		<edge from-layer="254" from-port="1" to-layer="256" to-port="0" />
		<edge from-layer="255" from-port="0" to-layer="256" to-port="1" />
		<edge from-layer="256" from-port="2" to-layer="258" to-port="0" />
		<edge from-layer="257" from-port="0" to-layer="258" to-port="1" />
		<edge from-layer="258" from-port="2" to-layer="259" to-port="0" />
		<edge from-layer="259" from-port="2" to-layer="261" to-port="0" />
		<edge from-layer="260" from-port="0" to-layer="261" to-port="1" />
		<edge from-layer="261" from-port="2" to-layer="263" to-port="0" />
		<edge from-layer="262" from-port="0" to-layer="263" to-port="1" />
		<edge from-layer="263" from-port="2" to-layer="264" to-port="0" />
		<edge from-layer="264" from-port="1" to-layer="266" to-port="0" />
		<edge from-layer="265" from-port="0" to-layer="266" to-port="1" />
		<edge from-layer="266" from-port="2" to-layer="268" to-port="0" />
		<edge from-layer="267" from-port="0" to-layer="268" to-port="1" />
		<edge from-layer="268" from-port="2" to-layer="269" to-port="0" />
		<edge from-layer="269" from-port="1" to-layer="271" to-port="0" />
		<edge from-layer="270" from-port="0" to-layer="271" to-port="1" />
		<edge from-layer="271" from-port="2" to-layer="273" to-port="0" />
		<edge from-layer="272" from-port="0" to-layer="273" to-port="1" />
		<edge from-layer="273" from-port="2" to-layer="312" to-port="0" />
		<edge from-layer="273" from-port="2" to-layer="359" to-port="0" />
		<edge from-layer="273" from-port="2" to-layer="275" to-port="0" />
		<edge from-layer="274" from-port="0" to-layer="275" to-port="1" />
		<edge from-layer="275" from-port="2" to-layer="277" to-port="0" />
		<edge from-layer="276" from-port="0" to-layer="277" to-port="1" />
		<edge from-layer="277" from-port="2" to-layer="278" to-port="0" />
		<edge from-layer="278" from-port="1" to-layer="280" to-port="0" />
		<edge from-layer="279" from-port="0" to-layer="280" to-port="1" />
		<edge from-layer="280" from-port="2" to-layer="282" to-port="0" />
		<edge from-layer="281" from-port="0" to-layer="282" to-port="1" />
		<edge from-layer="282" from-port="2" to-layer="284" to-port="0" />
		<edge from-layer="282" from-port="2" to-layer="285" to-port="0" />
		<edge from-layer="283" from-port="0" to-layer="284" to-port="1" />
		<edge from-layer="284" from-port="2" to-layer="291" to-port="0" />
		<edge from-layer="285" from-port="1" to-layer="288" to-port="0" />
		<edge from-layer="286" from-port="0" to-layer="288" to-port="1" />
		<edge from-layer="287" from-port="0" to-layer="288" to-port="2" />
		<edge from-layer="288" from-port="3" to-layer="290" to-port="0" />
		<edge from-layer="289" from-port="0" to-layer="290" to-port="1" />
		<edge from-layer="290" from-port="2" to-layer="291" to-port="1" />
		<edge from-layer="291" from-port="2" to-layer="292" to-port="1" />
		<edge from-layer="292" from-port="2" to-layer="372" to-port="0" />
		<edge from-layer="293" from-port="0" to-layer="294" to-port="1" />
		<edge from-layer="294" from-port="2" to-layer="296" to-port="0" />
		<edge from-layer="295" from-port="0" to-layer="296" to-port="1" />
		<edge from-layer="296" from-port="2" to-layer="297" to-port="0" />
		<edge from-layer="297" from-port="1" to-layer="299" to-port="0" />
		<edge from-layer="298" from-port="0" to-layer="299" to-port="1" />
		<edge from-layer="299" from-port="2" to-layer="301" to-port="0" />
		<edge from-layer="300" from-port="0" to-layer="301" to-port="1" />
		<edge from-layer="301" from-port="2" to-layer="303" to-port="0" />
		<edge from-layer="301" from-port="2" to-layer="304" to-port="0" />
		<edge from-layer="302" from-port="0" to-layer="303" to-port="1" />
		<edge from-layer="303" from-port="2" to-layer="310" to-port="0" />
		<edge from-layer="304" from-port="1" to-layer="307" to-port="0" />
		<edge from-layer="305" from-port="0" to-layer="307" to-port="1" />
		<edge from-layer="306" from-port="0" to-layer="307" to-port="2" />
		<edge from-layer="307" from-port="3" to-layer="309" to-port="0" />
		<edge from-layer="308" from-port="0" to-layer="309" to-port="1" />
		<edge from-layer="309" from-port="2" to-layer="310" to-port="1" />
		<edge from-layer="310" from-port="2" to-layer="329" to-port="0" />
		<edge from-layer="311" from-port="0" to-layer="312" to-port="1" />
		<edge from-layer="312" from-port="2" to-layer="314" to-port="0" />
		<edge from-layer="313" from-port="0" to-layer="314" to-port="1" />
		<edge from-layer="314" from-port="2" to-layer="315" to-port="0" />
		<edge from-layer="315" from-port="1" to-layer="317" to-port="0" />
		<edge from-layer="316" from-port="0" to-layer="317" to-port="1" />
		<edge from-layer="317" from-port="2" to-layer="319" to-port="0" />
		<edge from-layer="318" from-port="0" to-layer="319" to-port="1" />
		<edge from-layer="319" from-port="2" to-layer="321" to-port="0" />
		<edge from-layer="319" from-port="2" to-layer="322" to-port="0" />
		<edge from-layer="320" from-port="0" to-layer="321" to-port="1" />
		<edge from-layer="321" from-port="2" to-layer="328" to-port="0" />
		<edge from-layer="322" from-port="1" to-layer="325" to-port="0" />
		<edge from-layer="323" from-port="0" to-layer="325" to-port="1" />
		<edge from-layer="324" from-port="0" to-layer="325" to-port="2" />
		<edge from-layer="325" from-port="3" to-layer="327" to-port="0" />
		<edge from-layer="326" from-port="0" to-layer="327" to-port="1" />
		<edge from-layer="327" from-port="2" to-layer="328" to-port="1" />
		<edge from-layer="328" from-port="2" to-layer="329" to-port="1" />
		<edge from-layer="329" from-port="2" to-layer="331" to-port="0" />
		<edge from-layer="330" from-port="0" to-layer="331" to-port="1" />
		<edge from-layer="331" from-port="2" to-layer="332" to-port="0" />
		<edge from-layer="331" from-port="2" to-layer="341" to-port="0" />
		<edge from-layer="332" from-port="1" to-layer="343" to-port="1" />
		<edge from-layer="332" from-port="1" to-layer="336" to-port="0" />
		<edge from-layer="333" from-port="0" to-layer="336" to-port="1" />
		<edge from-layer="334" from-port="0" to-layer="336" to-port="2" />
		<edge from-layer="335" from-port="0" to-layer="336" to-port="3" />
		<edge from-layer="336" from-port="4" to-layer="338" to-port="0" />
		<edge from-layer="337" from-port="0" to-layer="338" to-port="1" />
		<edge from-layer="338" from-port="2" to-layer="340" to-port="0" />
		<edge from-layer="339" from-port="0" to-layer="340" to-port="1" />
		<edge from-layer="340" from-port="2" to-layer="341" to-port="1" />
		<edge from-layer="341" from-port="2" to-layer="342" to-port="0" />
		<edge from-layer="342" from-port="1" to-layer="343" to-port="0" />
		<edge from-layer="343" from-port="2" to-layer="345" to-port="0" />
		<edge from-layer="344" from-port="0" to-layer="345" to-port="1" />
		<edge from-layer="345" from-port="2" to-layer="372" to-port="1" />
		<edge from-layer="346" from-port="1" to-layer="350" to-port="0" />
		<edge from-layer="347" from-port="0" to-layer="350" to-port="1" />
		<edge from-layer="348" from-port="0" to-layer="350" to-port="2" />
		<edge from-layer="349" from-port="0" to-layer="350" to-port="3" />
		<edge from-layer="350" from-port="4" to-layer="356" to-port="0" />
		<edge from-layer="351" from-port="1" to-layer="355" to-port="0" />
		<edge from-layer="351" from-port="1" to-layer="367" to-port="0" />
		<edge from-layer="352" from-port="0" to-layer="355" to-port="1" />
		<edge from-layer="353" from-port="0" to-layer="355" to-port="2" />
		<edge from-layer="354" from-port="0" to-layer="355" to-port="3" />
		<edge from-layer="355" from-port="4" to-layer="356" to-port="1" />
		<edge from-layer="356" from-port="2" to-layer="358" to-port="0" />
		<edge from-layer="357" from-port="0" to-layer="358" to-port="1" />
		<edge from-layer="358" from-port="2" to-layer="371" to-port="0" />
		<edge from-layer="359" from-port="1" to-layer="363" to-port="0" />
		<edge from-layer="360" from-port="0" to-layer="363" to-port="1" />
		<edge from-layer="361" from-port="0" to-layer="363" to-port="2" />
		<edge from-layer="362" from-port="0" to-layer="363" to-port="3" />
		<edge from-layer="363" from-port="4" to-layer="368" to-port="0" />
		<edge from-layer="364" from-port="0" to-layer="367" to-port="1" />
		<edge from-layer="365" from-port="0" to-layer="367" to-port="2" />
		<edge from-layer="366" from-port="0" to-layer="367" to-port="3" />
		<edge from-layer="367" from-port="4" to-layer="368" to-port="1" />
		<edge from-layer="368" from-port="2" to-layer="370" to-port="0" />
		<edge from-layer="369" from-port="0" to-layer="370" to-port="1" />
		<edge from-layer="370" from-port="2" to-layer="371" to-port="1" />
		<edge from-layer="371" from-port="2" to-layer="372" to-port="2" />
		<edge from-layer="372" from-port="3" to-layer="373" to-port="0" />
	</edges>
	<rt_info>
		<MO_version value="custom_HEAD_f6ee6e92f846a8c665e4a7089c51481f9689a3b5" />
		<Runtime_version value="2023.0.0-10521-f6ee6e92f84-HEAD" />
		<conversion_parameters>
			<compress_to_fp16 value="False" />
			<framework value="onnx" />
			<input value="image" />
			<input_model value="DIR/model.onnx" />
			<input_shape value="[1,3,256,256]" />
			<layout value="image(nchw)" />
			<model_name value="person-vehicle-bike-detection-2000" />
			<output value="detection_out" />
			<output_dir value="DIR" />
			<reverse_input_channels value="True" />
			<scale_values value="image[255.0]" />
		</conversion_parameters>
		<legacy_frontend value="False" />
	</rt_info>
</net>
