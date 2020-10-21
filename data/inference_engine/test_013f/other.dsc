<?xml version="1.0" ?>
<net name="person-vehicle-bike-detection-crossroad-1016" version="10">
	<layers>
		<layer id="0" name="input.1" type="Parameter" version="opset1">
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
		<layer id="3" name="data_mul_1221112215/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4" shape="1,3,1,1" size="12"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="391/mean/Fused_Mul_" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
				</port>
				<port id="1">
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
					<dim>512</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="data_add_12217/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="16" shape="1,3,1,1" size="12"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="391/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>512</dim>
					<dim>512</dim>
				</port>
				<port id="1">
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
					<dim>512</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="393/mean/Fused_Mul_1332113323_const" type="Const" version="opset1">
			<data element_type="f32" offset="28" shape="32,3,3,3" size="3456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="392" type="Convolution" version="opset1">
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
		<layer id="9" name="data_add_1222012225/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3484" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="393/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="11" name="394" type="ReLU" version="opset1">
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
		<layer id="12" name="2046220465_const" type="Const" version="opset1">
			<data element_type="f32" offset="3612" shape="32,1,1,3,3" size="1152"/>
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
		<layer id="13" name="395" type="GroupConvolution" version="opset1">
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
		<layer id="14" name="data_add_1222812233/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4764" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="396/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="16" name="397" type="ReLU" version="opset1">
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
		<layer id="17" name="399/mean/Fused_Mul_1332913331_const" type="Const" version="opset1">
			<data element_type="f32" offset="4892" shape="16,32,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>16</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="398" type="Convolution" version="opset1">
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
		<layer id="19" name="data_add_1223612241/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="6940" shape="1,16,1,1" size="64"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="399/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="21" name="401/mean/Fused_Mul_1333313335_const" type="Const" version="opset1">
			<data element_type="f32" offset="7004" shape="96,16,1,1" size="6144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>16</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="400" type="Convolution" version="opset1">
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
		<layer id="23" name="data_add_1224412249/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="13148" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="401/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="25" name="402" type="ReLU" version="opset1">
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
		<layer id="26" name="2049020493_const" type="Const" version="opset1">
			<data element_type="f32" offset="13532" shape="96,1,1,3,3" size="3456"/>
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
		<layer id="27" name="403" type="GroupConvolution" version="opset1">
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
		<layer id="28" name="data_add_1225212257/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="16988" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="404/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="30" name="405" type="ReLU" version="opset1">
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
		<layer id="31" name="407/mean/Fused_Mul_1334113343_const" type="Const" version="opset1">
			<data element_type="f32" offset="17372" shape="24,96,1,1" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="406" type="Convolution" version="opset1">
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
		<layer id="33" name="data_add_1226012265/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="26588" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="407/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="35" name="409/mean/Fused_Mul_1334513347_const" type="Const" version="opset1">
			<data element_type="f32" offset="26684" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="408" type="Convolution" version="opset1">
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
		<layer id="37" name="data_add_1226812273/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="40508" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="409/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="39" name="410" type="ReLU" version="opset1">
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
		<layer id="40" name="2048220485_const" type="Const" version="opset1">
			<data element_type="f32" offset="41084" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="41" name="411" type="GroupConvolution" version="opset1">
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
		<layer id="42" name="data_add_1227612281/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="46268" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="412/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="44" name="413" type="ReLU" version="opset1">
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
		<layer id="45" name="415/mean/Fused_Mul_1335313355_const" type="Const" version="opset1">
			<data element_type="f32" offset="46844" shape="24,144,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>24</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="414" type="Convolution" version="opset1">
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
		<layer id="47" name="data_add_1228412289/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="60668" shape="1,24,1,1" size="96"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="415/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="49" name="416" type="Add" version="opset1">
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
		<layer id="50" name="418/mean/Fused_Mul_1335713359_const" type="Const" version="opset1">
			<data element_type="f32" offset="60764" shape="144,24,1,1" size="13824"/>
			<output>
				<port id="1" precision="FP32">
					<dim>144</dim>
					<dim>24</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="417" type="Convolution" version="opset1">
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
		<layer id="52" name="data_add_1229212297/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="74588" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="418/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="54" name="419" type="ReLU" version="opset1">
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
		<layer id="55" name="2045020453_const" type="Const" version="opset1">
			<data element_type="f32" offset="75164" shape="144,1,1,3,3" size="5184"/>
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
		<layer id="56" name="420" type="GroupConvolution" version="opset1">
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
		<layer id="57" name="data_add_1230012305/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="80348" shape="1,144,1,1" size="576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="421/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="59" name="422" type="ReLU" version="opset1">
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
		<layer id="60" name="424/mean/Fused_Mul_1336513367_const" type="Const" version="opset1">
			<data element_type="f32" offset="80924" shape="32,144,1,1" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>144</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="61" name="423" type="Convolution" version="opset1">
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
		<layer id="62" name="data_add_1230812313/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="99356" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="63" name="424/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="64" name="426/mean/Fused_Mul_1336913371_const" type="Const" version="opset1">
			<data element_type="f32" offset="99484" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="65" name="425" type="Convolution" version="opset1">
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
		<layer id="66" name="data_add_1231612321/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="124060" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="426/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="68" name="427" type="ReLU" version="opset1">
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
		<layer id="69" name="2046620469_const" type="Const" version="opset1">
			<data element_type="f32" offset="124828" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="70" name="428" type="GroupConvolution" version="opset1">
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
		<layer id="71" name="data_add_1232412329/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="131740" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="429/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="73" name="430" type="ReLU" version="opset1">
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
		<layer id="74" name="432/mean/Fused_Mul_1337713379_const" type="Const" version="opset1">
			<data element_type="f32" offset="132508" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="75" name="431" type="Convolution" version="opset1">
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
		<layer id="76" name="data_add_1233212337/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="157084" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="77" name="432/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="78" name="433" type="Add" version="opset1">
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
		<layer id="79" name="435/mean/Fused_Mul_1338113383_const" type="Const" version="opset1">
			<data element_type="f32" offset="157212" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="80" name="434" type="Convolution" version="opset1">
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
		<layer id="81" name="data_add_1234012345/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="181788" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="82" name="435/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="83" name="436" type="ReLU" version="opset1">
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
		<layer id="84" name="2045820461_const" type="Const" version="opset1">
			<data element_type="f32" offset="182556" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="85" name="437" type="GroupConvolution" version="opset1">
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
		<layer id="86" name="data_add_1234812353/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="189468" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="87" name="438/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="88" name="439" type="ReLU" version="opset1">
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
		<layer id="89" name="441/mean/Fused_Mul_1338913391_const" type="Const" version="opset1">
			<data element_type="f32" offset="190236" shape="32,192,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>32</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="90" name="440" type="Convolution" version="opset1">
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
		<layer id="91" name="data_add_1235612361/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="214812" shape="1,32,1,1" size="128"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="92" name="441/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="93" name="442" type="Add" version="opset1">
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
		<layer id="94" name="444/mean/Fused_Mul_1339313395_const" type="Const" version="opset1">
			<data element_type="f32" offset="214940" shape="192,32,1,1" size="24576"/>
			<output>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>32</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="95" name="443" type="Convolution" version="opset1">
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
		<layer id="96" name="data_add_1236412369/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="239516" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="97" name="444/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="98" name="445" type="ReLU" version="opset1">
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
		<layer id="99" name="2043820441_const" type="Const" version="opset1">
			<data element_type="f32" offset="240284" shape="192,1,1,3,3" size="6912"/>
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
		<layer id="100" name="446" type="GroupConvolution" version="opset1">
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
		<layer id="101" name="data_add_1237212377/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="247196" shape="1,192,1,1" size="768"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="102" name="447/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="103" name="448" type="ReLU" version="opset1">
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
		<layer id="104" name="450/mean/Fused_Mul_1340113403_const" type="Const" version="opset1">
			<data element_type="f32" offset="247964" shape="64,192,1,1" size="49152"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="105" name="449" type="Convolution" version="opset1">
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
		<layer id="106" name="data_add_1238012385/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="297116" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="107" name="450/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="108" name="452/mean/Fused_Mul_1340513407_const" type="Const" version="opset1">
			<data element_type="f32" offset="297372" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="109" name="451" type="Convolution" version="opset1">
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
		<layer id="110" name="data_add_1238812393/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="395676" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="111" name="452/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="112" name="453" type="ReLU" version="opset1">
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
		<layer id="113" name="2047420477_const" type="Const" version="opset1">
			<data element_type="f32" offset="397212" shape="384,1,1,3,3" size="13824"/>
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
		<layer id="114" name="454" type="GroupConvolution" version="opset1">
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
		<layer id="115" name="data_add_1239612401/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="411036" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="116" name="455/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="117" name="456" type="ReLU" version="opset1">
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
		<layer id="118" name="458/mean/Fused_Mul_1341313415_const" type="Const" version="opset1">
			<data element_type="f32" offset="412572" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="119" name="457" type="Convolution" version="opset1">
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
		<layer id="120" name="data_add_1240412409/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="510876" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="121" name="458/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="122" name="459" type="Add" version="opset1">
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
		<layer id="123" name="461/mean/Fused_Mul_1341713419_const" type="Const" version="opset1">
			<data element_type="f32" offset="511132" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="124" name="460" type="Convolution" version="opset1">
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
		<layer id="125" name="data_add_1241212417/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="609436" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="126" name="461/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="127" name="462" type="ReLU" version="opset1">
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
		<layer id="128" name="2051020513_const" type="Const" version="opset1">
			<data element_type="f32" offset="610972" shape="384,1,1,3,3" size="13824"/>
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
		<layer id="129" name="463" type="GroupConvolution" version="opset1">
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
		<layer id="130" name="data_add_1242012425/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="624796" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="131" name="464/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="132" name="465" type="ReLU" version="opset1">
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
		<layer id="133" name="467/mean/Fused_Mul_1342513427_const" type="Const" version="opset1">
			<data element_type="f32" offset="626332" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="134" name="466" type="Convolution" version="opset1">
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
		<layer id="135" name="data_add_1242812433/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="724636" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="467/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="137" name="468" type="Add" version="opset1">
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
		<layer id="138" name="470/mean/Fused_Mul_1342913431_const" type="Const" version="opset1">
			<data element_type="f32" offset="724892" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="139" name="469" type="Convolution" version="opset1">
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
		<layer id="140" name="data_add_1243612441/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="823196" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="141" name="470/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="142" name="471" type="ReLU" version="opset1">
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
		<layer id="143" name="2049420497_const" type="Const" version="opset1">
			<data element_type="f32" offset="824732" shape="384,1,1,3,3" size="13824"/>
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
		<layer id="144" name="472" type="GroupConvolution" version="opset1">
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
		<layer id="145" name="data_add_1244412449/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="838556" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="146" name="473/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="147" name="474" type="ReLU" version="opset1">
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
		<layer id="148" name="476/mean/Fused_Mul_1343713439_const" type="Const" version="opset1">
			<data element_type="f32" offset="840092" shape="64,384,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="475" type="Convolution" version="opset1">
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
		<layer id="150" name="data_add_1245212457/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="938396" shape="1,64,1,1" size="256"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="151" name="476/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="152" name="477" type="Add" version="opset1">
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
		<layer id="153" name="479/mean/Fused_Mul_1344113443_const" type="Const" version="opset1">
			<data element_type="f32" offset="938652" shape="384,64,1,1" size="98304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>384</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="154" name="478" type="Convolution" version="opset1">
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
		<layer id="155" name="data_add_1246012465/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1036956" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="156" name="479/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="157" name="480" type="ReLU" version="opset1">
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
		<layer id="158" name="2051420517_const" type="Const" version="opset1">
			<data element_type="f32" offset="1038492" shape="384,1,1,3,3" size="13824"/>
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
		<layer id="159" name="481" type="GroupConvolution" version="opset1">
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
		<layer id="160" name="data_add_1246812473/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1052316" shape="1,384,1,1" size="1536"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="161" name="482/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="162" name="483" type="ReLU" version="opset1">
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
		<layer id="163" name="485/mean/Fused_Mul_1344913451_const" type="Const" version="opset1">
			<data element_type="f32" offset="1053852" shape="96,384,1,1" size="147456"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>384</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="164" name="484" type="Convolution" version="opset1">
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
		<layer id="165" name="data_add_1247612481/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1201308" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="166" name="485/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="167" name="487/mean/Fused_Mul_1345313455_const" type="Const" version="opset1">
			<data element_type="f32" offset="1201692" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="168" name="486" type="Convolution" version="opset1">
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
		<layer id="169" name="data_add_1248412489/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1422876" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="170" name="487/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="171" name="488" type="ReLU" version="opset1">
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
		<layer id="172" name="2050220505_const" type="Const" version="opset1">
			<data element_type="f32" offset="1425180" shape="576,1,1,3,3" size="20736"/>
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
		<layer id="173" name="489" type="GroupConvolution" version="opset1">
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
		<layer id="174" name="data_add_1249212497/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1445916" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="175" name="490/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="176" name="491" type="ReLU" version="opset1">
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
		<layer id="177" name="493/mean/Fused_Mul_1346113463_const" type="Const" version="opset1">
			<data element_type="f32" offset="1448220" shape="96,576,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="178" name="492" type="Convolution" version="opset1">
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
		<layer id="179" name="data_add_1250012505/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1669404" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="180" name="493/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="181" name="494" type="Add" version="opset1">
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
		<layer id="182" name="496/mean/Fused_Mul_1346513467_const" type="Const" version="opset1">
			<data element_type="f32" offset="1669788" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="183" name="495" type="Convolution" version="opset1">
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
		<layer id="184" name="data_add_1250812513/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1890972" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="185" name="496/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="186" name="497" type="ReLU" version="opset1">
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
		<layer id="187" name="2043420437_const" type="Const" version="opset1">
			<data element_type="f32" offset="1893276" shape="576,1,1,3,3" size="20736"/>
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
		<layer id="188" name="498" type="GroupConvolution" version="opset1">
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
		<layer id="189" name="data_add_1251612521/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="1914012" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="190" name="499/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="191" name="500" type="ReLU" version="opset1">
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
		<layer id="192" name="502/mean/Fused_Mul_1347313475_const" type="Const" version="opset1">
			<data element_type="f32" offset="1916316" shape="96,576,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="193" name="501" type="Convolution" version="opset1">
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
		<layer id="194" name="data_add_1252412529/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2137500" shape="1,96,1,1" size="384"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="195" name="502/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="196" name="503" type="Add" version="opset1">
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
		<layer id="197" name="505/mean/Fused_Mul_1347713479_const" type="Const" version="opset1">
			<data element_type="f32" offset="2137884" shape="576,96,1,1" size="221184"/>
			<output>
				<port id="1" precision="FP32">
					<dim>576</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="198" name="504" type="Convolution" version="opset1">
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
		<layer id="199" name="data_add_1253212537/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2359068" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="200" name="505/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="201" name="506" type="ReLU" version="opset1">
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
		<layer id="202" name="2051820521_const" type="Const" version="opset1">
			<data element_type="f32" offset="2361372" shape="576,1,1,3,3" size="20736"/>
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
		<layer id="203" name="554/WithoutBiases" type="GroupConvolution" version="opset1">
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
		<layer id="204" name="data_add_1254012545/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2382108" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="205" name="554/Fused_Add_" type="Add" version="opset1">
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
		<layer id="206" name="556" type="ReLU" version="opset1">
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
		<layer id="207" name="onnx_initializer_node_343/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="2384412" shape="36,576,1,1" size="82944"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="208" name="557/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="209" name="557/Dims7379/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2467356" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="210" name="557" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="211" name="625/Cast_121005_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="212" name="625" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="213" name="627" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="214" name="6281113/Cast_120985_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="215" name="6281113/Cast_220987_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="216" name="6281113" type="Gather" version="opset1">
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
		<layer id="217" name="630/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="218" name="630/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="219" name="631/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="220" name="632" type="Concat" version="opset1">
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
		<layer id="221" name="633" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36864</dim>
				</port>
			</output>
		</layer>
		<layer id="222" name="2044220445_const" type="Const" version="opset1">
			<data element_type="f32" offset="2467552" shape="576,1,1,3,3" size="20736"/>
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
		<layer id="223" name="507" type="GroupConvolution" version="opset1">
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
		<layer id="224" name="data_add_1255612561/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2488288" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="225" name="508/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="226" name="509" type="ReLU" version="opset1">
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
		<layer id="227" name="511/mean/Fused_Mul_1349313495_const" type="Const" version="opset1">
			<data element_type="f32" offset="2490592" shape="160,576,1,1" size="368640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="228" name="510" type="Convolution" version="opset1">
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
		<layer id="229" name="data_add_1256412569/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="2859232" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="230" name="511/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="231" name="513/mean/Fused_Mul_1349713499_const" type="Const" version="opset1">
			<data element_type="f32" offset="2859872" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="232" name="512" type="Convolution" version="opset1">
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
		<layer id="233" name="data_add_1257212577/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3474272" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="234" name="513/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="235" name="514" type="ReLU" version="opset1">
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
		<layer id="236" name="2048620489_const" type="Const" version="opset1">
			<data element_type="f32" offset="3478112" shape="960,1,1,3,3" size="34560"/>
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
		<layer id="237" name="515" type="GroupConvolution" version="opset1">
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
		<layer id="238" name="data_add_1258012585/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="3512672" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="239" name="516/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="240" name="517" type="ReLU" version="opset1">
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
		<layer id="241" name="519/mean/Fused_Mul_1350513507_const" type="Const" version="opset1">
			<data element_type="f32" offset="3516512" shape="160,960,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="242" name="518" type="Convolution" version="opset1">
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
		<layer id="243" name="data_add_1258812593/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4130912" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="244" name="519/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="245" name="520" type="Add" version="opset1">
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
		<layer id="246" name="522/mean/Fused_Mul_1350913511_const" type="Const" version="opset1">
			<data element_type="f32" offset="4131552" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="247" name="521" type="Convolution" version="opset1">
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
		<layer id="248" name="data_add_1259612601/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4745952" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="249" name="522/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="250" name="523" type="ReLU" version="opset1">
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
		<layer id="251" name="2047020473_const" type="Const" version="opset1">
			<data element_type="f32" offset="4749792" shape="960,1,1,3,3" size="34560"/>
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
		<layer id="252" name="524" type="GroupConvolution" version="opset1">
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
		<layer id="253" name="data_add_1260412609/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="4784352" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="254" name="525/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="255" name="526" type="ReLU" version="opset1">
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
		<layer id="256" name="528/mean/Fused_Mul_1351713519_const" type="Const" version="opset1">
			<data element_type="f32" offset="4788192" shape="160,960,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>160</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="257" name="527" type="Convolution" version="opset1">
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
		<layer id="258" name="data_add_1261212617/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="5402592" shape="1,160,1,1" size="640"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="259" name="528/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="260" name="529" type="Add" version="opset1">
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
		<layer id="261" name="531/mean/Fused_Mul_1352113523_const" type="Const" version="opset1">
			<data element_type="f32" offset="5403232" shape="960,160,1,1" size="614400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>960</dim>
					<dim>160</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="262" name="530" type="Convolution" version="opset1">
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
		<layer id="263" name="data_add_1262012625/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="6017632" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="264" name="531/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="265" name="532" type="ReLU" version="opset1">
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
		<layer id="266" name="2050620509_const" type="Const" version="opset1">
			<data element_type="f32" offset="6021472" shape="960,1,1,3,3" size="34560"/>
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
		<layer id="267" name="533" type="GroupConvolution" version="opset1">
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
		<layer id="268" name="data_add_1262812633/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="6056032" shape="1,960,1,1" size="3840"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="269" name="534/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="270" name="535" type="ReLU" version="opset1">
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
		<layer id="271" name="537/mean/Fused_Mul_1352913531_const" type="Const" version="opset1">
			<data element_type="f32" offset="6059872" shape="320,960,1,1" size="1228800"/>
			<output>
				<port id="1" precision="FP32">
					<dim>320</dim>
					<dim>960</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="272" name="536" type="Convolution" version="opset1">
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
		<layer id="273" name="data_add_1263612641/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="7288672" shape="1,320,1,1" size="1280"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="274" name="537/variance/Fused_Add_" type="Add" version="opset1">
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
		<layer id="275" name="539/mean/Fused_Mul_1353313535_const" type="Const" version="opset1">
			<data element_type="f32" offset="7289952" shape="1280,320,1,1" size="1638400"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1280</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="276" name="538" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>320</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1280</dim>
					<dim>320</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="277" name="data_add_1264412649/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8928352" shape="1,1280,1,1" size="5120"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="278" name="539/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="279" name="540" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="280" name="2045420457_const" type="Const" version="opset1">
			<data element_type="f32" offset="8933472" shape="1280,1,1,3,3" size="46080"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="281" name="562/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="282" name="data_add_1265212657/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="8979552" shape="1,1280,1,1" size="5120"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="283" name="562/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="284" name="564" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="285" name="onnx_initializer_node_352/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="8984672" shape="36,1280,1,1" size="184320"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="286" name="565/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="287" name="565/Dims7355/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="9168992" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="288" name="565" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="289" name="634/Cast_121007_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="290" name="634" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="291" name="636" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="292" name="6371105/Cast_120995_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="293" name="6371105/Cast_220997_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="294" name="6371105" type="Gather" version="opset1">
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
		<layer id="295" name="639/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="296" name="639/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="297" name="640/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="298" name="641" type="Concat" version="opset1">
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
		<layer id="299" name="642" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9216</dim>
				</port>
			</output>
		</layer>
		<layer id="300" name="542/mean/Fused_Mul_1354513547_const" type="Const" version="opset1">
			<data element_type="f32" offset="9169136" shape="256,1280,1,1" size="1310720"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="301" name="541" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="302" name="data_add_1266812673/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="10479856" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="303" name="542/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
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
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="304" name="543" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="305" name="2049820501_const" type="Const" version="opset1">
			<data element_type="f32" offset="10480880" shape="256,1,1,3,3" size="9216"/>
			<output>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="306" name="544" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="307" name="data_add_1267612681/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="10490096" shape="1,256,1,1" size="1024"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="308" name="545/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="309" name="546" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="310" name="548/mean/Fused_Mul_1355313555_const" type="Const" version="opset1">
			<data element_type="f32" offset="10491120" shape="512,256,1,1" size="524288"/>
			<output>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="311" name="547" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>8</dim>
					<dim>8</dim>
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="312" name="data_add_1268412689/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11015408" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="313" name="548/variance/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="314" name="549" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="315" name="2044620449_const" type="Const" version="opset1">
			<data element_type="f32" offset="11017456" shape="512,1,1,3,3" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="316" name="570/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="317" name="data_add_1269212697/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11035888" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="318" name="570/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="319" name="572" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="320" name="onnx_initializer_node_361/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="11037936" shape="36,512,1,1" size="73728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="321" name="573/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="322" name="573/Dims7361/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11111664" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="323" name="573" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="324" name="643/Cast_120931_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="325" name="643" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="326" name="645" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="327" name="6461107/Cast_121009_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="328" name="6461107/Cast_221011_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="329" name="6461107" type="Gather" version="opset1">
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
		<layer id="330" name="648/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="331" name="648/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="332" name="649/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="333" name="650" type="Concat" version="opset1">
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
		<layer id="334" name="651" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2304</dim>
				</port>
			</output>
		</layer>
		<layer id="335" name="652" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36864</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>9216</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2304</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
			</output>
		</layer>
		<layer id="336" name="2052220525_const" type="Const" version="opset1">
			<data element_type="f32" offset="11111808" shape="576,1,1,3,3" size="20736"/>
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
		<layer id="337" name="550/WithoutBiases" type="GroupConvolution" version="opset1">
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
		<layer id="338" name="data_add_1254812553/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11132544" shape="1,576,1,1" size="2304"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="339" name="550/Fused_Add_" type="Add" version="opset1">
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
		<layer id="340" name="552" type="ReLU" version="opset1">
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
		<layer id="341" name="onnx_initializer_node_370/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="11134848" shape="36,576,1,1" size="82944"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="342" name="553/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>576</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="343" name="553/Dims7325/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11217792" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="344" name="553" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="345" name="578/Cast_120947_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="346" name="578" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="347" name="580" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="348" name="5811111/Cast_120915_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="349" name="5811111/Cast_220917_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="350" name="5811111" type="Gather" version="opset1">
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
		<layer id="351" name="583/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="352" name="583/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="353" name="584/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="354" name="585" type="Concat" version="opset1">
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
		<layer id="355" name="586" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36864</dim>
				</port>
			</output>
		</layer>
		<layer id="356" name="2047820481_const" type="Const" version="opset1">
			<data element_type="f32" offset="11217936" shape="1280,1,1,3,3" size="46080"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="357" name="558/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="358" name="data_add_1266012665/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11264016" shape="1,1280,1,1" size="5120"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="359" name="558/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="360" name="560" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="361" name="onnx_initializer_node_379/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="11269136" shape="36,1280,1,1" size="184320"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="362" name="561/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>1280</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="363" name="561/Dims7343/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11453456" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="364" name="561" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="365" name="587/Cast_120939_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="366" name="587" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="367" name="589" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="368" name="5901101/Cast_120943_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="369" name="5901101/Cast_220945_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="370" name="5901101" type="Gather" version="opset1">
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
		<layer id="371" name="592/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="372" name="592/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="373" name="593/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="374" name="594" type="Concat" version="opset1">
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
		<layer id="375" name="595" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>16</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>9216</dim>
				</port>
			</output>
		</layer>
		<layer id="376" name="2052620529_const" type="Const" version="opset1">
			<data element_type="f32" offset="11453600" shape="512,1,1,3,3" size="18432"/>
			<output>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="377" name="566/WithoutBiases" type="GroupConvolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="378" name="data_add_1270012705/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11472032" shape="1,512,1,1" size="2048"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="379" name="566/Fused_Add_" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
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
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="380" name="568" type="ReLU" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="381" name="onnx_initializer_node_388/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="11474080" shape="36,512,1,1" size="73728"/>
			<output>
				<port id="1" precision="FP32">
					<dim>36</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="382" name="569/WithoutBiases" type="Convolution" version="opset1">
			<data dilations="1,1" output_padding="0,0" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>36</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="383" name="569/Dims7337/copy_const" type="Const" version="opset1">
			<data element_type="f32" offset="11547808" shape="1,36,1,1" size="144"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="384" name="569" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>36</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="385" name="596/Cast_121017_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467500" shape="4" size="32"/>
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="386" name="596" type="Transpose" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>36</dim>
				</port>
			</output>
		</layer>
		<layer id="387" name="598" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36</dim>
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
		<layer id="388" name="5991109/Cast_120979_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="389" name="5991109/Cast_220981_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="390" name="5991109" type="Gather" version="opset1">
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
		<layer id="391" name="601/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="392" name="601/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="393" name="602/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="394" name="603" type="Concat" version="opset1">
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
		<layer id="395" name="604" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
					<dim>36</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2304</dim>
				</port>
			</output>
		</layer>
		<layer id="396" name="605" type="Concat" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>36864</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>9216</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2304</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
			</output>
		</layer>
		<layer id="397" name="607" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="398" name="6081115/Cast_120953_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="399" name="6081115/Cast_220955_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="400" name="6081115" type="Gather" version="opset1">
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
		<layer id="401" name="611/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="402" name="611/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="403" name="612/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="404" name="613/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="405" name="614" type="Concat" version="opset1">
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
		<layer id="406" name="615" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="407" name="616/FlattenONNX_/input_shape" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="408" name="616/FlattenONNX_/input_shape/Gather/Cast_120949_const" type="Const" version="opset1">
			<data element_type="i32" offset="11547960" shape="2" size="8"/>
			<output>
				<port id="1" precision="I32">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="409" name="616/FlattenONNX_/input_shape/Gather/Cast_220951_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="410" name="616/FlattenONNX_/input_shape/Gather" type="Gather" version="opset1">
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
		<layer id="411" name="616/FlattenONNX_/first_dims/Cast_120919_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="412" name="616/FlattenONNX_/first_dims" type="ReduceProd" version="opset1">
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
		<layer id="413" name="616/FlattenONNX_/second_dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="414" name="616/FlattenONNX_/first_dims/shapes_concat" type="Concat" version="opset1">
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
		<layer id="415" name="616/FlattenONNX_/Reshape" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="416" name="616/Softmax_" type="SoftMax" version="opset1">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="417" name="616/ShapeOf_" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="418" name="616" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>12096</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="419" name="618" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="420" name="6191103/Cast_120991_const" type="Const" version="opset1">
			<data element_type="i32" offset="2467532" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="421" name="6191103/Cast_220993_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="" size="8"/>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="422" name="6191103" type="Gather" version="opset1">
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
		<layer id="423" name="621/Dims/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="424" name="621/Unsqueeze" type="Unsqueeze" version="opset1">
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
		<layer id="425" name="622/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="i64" offset="2467544" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="426" name="623" type="Concat" version="opset1">
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
		<layer id="427" name="624" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>12096</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
			</output>
		</layer>
		<layer id="428" name="574/0_port" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>576</dim>
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
		<layer id="429" name="574/ss_0_port/Cast_120957_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="430" name="574/ss_0_port/Cast_220959_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="431" name="574/ss_0_port/Cast_320961_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="432" name="574/ss_0_port" type="StridedSlice" version="opset1">
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
		<layer id="433" name="574/1_port" type="ShapeOf" version="opset3">
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
		<layer id="434" name="574/ss_1_port/Cast_120999_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="435" name="574/ss_1_port/Cast_221001_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="436" name="574/ss_1_port/Cast_321003_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="437" name="574/ss_1_port" type="StridedSlice" version="opset1">
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
		<layer id="438" name="574/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="20.733,45.464,78.592,29.393,55.398,84.88,17.006,28.673,44.11" img_h="0" img_size="0" img_w="0" offset="0.5" step="16.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="17.137,38.165,70.69,9.584,17.634,23.744,6.507,12.245,14.749"/>
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
					<dim>36864</dim>
				</port>
			</output>
		</layer>
		<layer id="439" name="574/unsqueeze/value15740_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="440" name="574" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>36864</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>36864</dim>
				</port>
			</output>
		</layer>
		<layer id="441" name="575/0_port" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1280</dim>
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
		<layer id="442" name="575/ss_0_port/Cast_120933_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="443" name="575/ss_0_port/Cast_220935_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="444" name="575/ss_0_port/Cast_320937_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="445" name="575/ss_0_port" type="StridedSlice" version="opset1">
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
		<layer id="446" name="575/1_port" type="ShapeOf" version="opset3">
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
		<layer id="447" name="575/ss_1_port/Cast_120921_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="448" name="575/ss_1_port/Cast_220923_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="449" name="575/ss_1_port/Cast_320925_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="450" name="575/ss_1_port" type="StridedSlice" version="opset1">
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
		<layer id="451" name="575/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="157.379,104.698,210.545,118.319,157.328,203.363,36.256,64.451,101.718" img_h="0" img_size="0" img_w="0" offset="0.5" step="32.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="81.753,153.183,169.567,32.148,41.048,52.198,32.391,22.397,33.216"/>
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
					<dim>9216</dim>
				</port>
			</output>
		</layer>
		<layer id="452" name="575/unsqueeze/value15758_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="453" name="575" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>9216</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>9216</dim>
				</port>
			</output>
		</layer>
		<layer id="454" name="576/0_port" type="ShapeOf" version="opset3">
			<data output_type="i64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
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
		<layer id="455" name="576/ss_0_port/Cast_120963_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="456" name="576/ss_0_port/Cast_220965_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="457" name="576/ss_0_port/Cast_320967_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="458" name="576/ss_0_port" type="StridedSlice" version="opset1">
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
		<layer id="459" name="576/1_port" type="ShapeOf" version="opset3">
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
		<layer id="460" name="576/ss_1_port/Cast_120971_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547968" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="461" name="576/ss_1_port/Cast_220973_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547952" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="462" name="576/ss_1_port/Cast_320975_const" type="Const" version="opset1">
			<data element_type="i64" offset="11547976" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="463" name="576/ss_1_port" type="StridedSlice" version="opset1">
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
		<layer id="464" name="576/naked_not_unsqueezed" type="PriorBoxClustered" version="opset1">
			<data clip="0" flip="0" height="344.064,243.971,337.749,256.941,327.187,428.114,68.919,155.867,270.048" img_h="0" img_size="0" img_w="0" offset="0.5" step="64.0" step_h="0.0" step_w="0.0" variance="0.1,0.1,0.2,0.2" width="110.651,237.237,348.269,65.598,82.729,110.538,53.24,68.246,105.444"/>
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
					<dim>2304</dim>
				</port>
			</output>
		</layer>
		<layer id="465" name="576/unsqueeze/value15776_const" type="Const" version="opset1">
			<data element_type="i64" offset="2467536" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="466" name="576" type="Unsqueeze" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>2304</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2304</dim>
				</port>
			</output>
		</layer>
		<layer id="467" name="577" type="Concat" version="opset1">
			<data axis="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>36864</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>9216</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2304</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48384</dim>
				</port>
			</output>
		</layer>
		<layer id="468" name="653" type="DetectionOutput" version="opset1">
			<data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.019999999552965164" eta="1.0" height="0" height_scale="0" input_height="1" input_width="1" interp_mode="" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="1" num_classes="4" pad_mode="" pad_value="" prob="0" resize_mode="" share_location="1" top_k="200" variance_encoded_in_target="0" visualize_threshold="0.6" width="0" width_scale="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>48384</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>48384</dim>
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
		<layer id="469" name="653/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="6" from-port="2" to-layer="8" to-port="0"/>
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
		<edge from-layer="20" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="1" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="24" to-port="0"/>
		<edge from-layer="23" from-port="1" to-layer="24" to-port="1"/>
		<edge from-layer="24" from-port="2" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="27" to-port="0"/>
		<edge from-layer="26" from-port="1" to-layer="27" to-port="1"/>
		<edge from-layer="27" from-port="2" to-layer="29" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="29" to-port="1"/>
		<edge from-layer="29" from-port="2" to-layer="30" to-port="0"/>
		<edge from-layer="30" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="31" from-port="1" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="34" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="1"/>
		<edge from-layer="34" from-port="2" to-layer="36" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="1"/>
		<edge from-layer="36" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="38" to-port="1"/>
		<edge from-layer="38" from-port="2" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="41" to-port="0"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="1"/>
		<edge from-layer="41" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="43" to-port="1"/>
		<edge from-layer="43" from-port="2" to-layer="44" to-port="0"/>
		<edge from-layer="44" from-port="1" to-layer="46" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="46" to-port="1"/>
		<edge from-layer="46" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="47" from-port="1" to-layer="48" to-port="1"/>
		<edge from-layer="34" from-port="2" to-layer="49" to-port="0"/>
		<edge from-layer="48" from-port="2" to-layer="49" to-port="1"/>
		<edge from-layer="49" from-port="2" to-layer="51" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="51" to-port="1"/>
		<edge from-layer="51" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="52" from-port="1" to-layer="53" to-port="1"/>
		<edge from-layer="53" from-port="2" to-layer="54" to-port="0"/>
		<edge from-layer="54" from-port="1" to-layer="56" to-port="0"/>
		<edge from-layer="55" from-port="1" to-layer="56" to-port="1"/>
		<edge from-layer="56" from-port="2" to-layer="58" to-port="0"/>
		<edge from-layer="57" from-port="1" to-layer="58" to-port="1"/>
		<edge from-layer="58" from-port="2" to-layer="59" to-port="0"/>
		<edge from-layer="59" from-port="1" to-layer="61" to-port="0"/>
		<edge from-layer="60" from-port="1" to-layer="61" to-port="1"/>
		<edge from-layer="61" from-port="2" to-layer="63" to-port="0"/>
		<edge from-layer="62" from-port="1" to-layer="63" to-port="1"/>
		<edge from-layer="63" from-port="2" to-layer="65" to-port="0"/>
		<edge from-layer="64" from-port="1" to-layer="65" to-port="1"/>
		<edge from-layer="65" from-port="2" to-layer="67" to-port="0"/>
		<edge from-layer="66" from-port="1" to-layer="67" to-port="1"/>
		<edge from-layer="67" from-port="2" to-layer="68" to-port="0"/>
		<edge from-layer="68" from-port="1" to-layer="70" to-port="0"/>
		<edge from-layer="69" from-port="1" to-layer="70" to-port="1"/>
		<edge from-layer="70" from-port="2" to-layer="72" to-port="0"/>
		<edge from-layer="71" from-port="1" to-layer="72" to-port="1"/>
		<edge from-layer="72" from-port="2" to-layer="73" to-port="0"/>
		<edge from-layer="73" from-port="1" to-layer="75" to-port="0"/>
		<edge from-layer="74" from-port="1" to-layer="75" to-port="1"/>
		<edge from-layer="75" from-port="2" to-layer="77" to-port="0"/>
		<edge from-layer="76" from-port="1" to-layer="77" to-port="1"/>
		<edge from-layer="63" from-port="2" to-layer="78" to-port="0"/>
		<edge from-layer="77" from-port="2" to-layer="78" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="80" to-port="0"/>
		<edge from-layer="79" from-port="1" to-layer="80" to-port="1"/>
		<edge from-layer="80" from-port="2" to-layer="82" to-port="0"/>
		<edge from-layer="81" from-port="1" to-layer="82" to-port="1"/>
		<edge from-layer="82" from-port="2" to-layer="83" to-port="0"/>
		<edge from-layer="83" from-port="1" to-layer="85" to-port="0"/>
		<edge from-layer="84" from-port="1" to-layer="85" to-port="1"/>
		<edge from-layer="85" from-port="2" to-layer="87" to-port="0"/>
		<edge from-layer="86" from-port="1" to-layer="87" to-port="1"/>
		<edge from-layer="87" from-port="2" to-layer="88" to-port="0"/>
		<edge from-layer="88" from-port="1" to-layer="90" to-port="0"/>
		<edge from-layer="89" from-port="1" to-layer="90" to-port="1"/>
		<edge from-layer="90" from-port="2" to-layer="92" to-port="0"/>
		<edge from-layer="91" from-port="1" to-layer="92" to-port="1"/>
		<edge from-layer="78" from-port="2" to-layer="93" to-port="0"/>
		<edge from-layer="92" from-port="2" to-layer="93" to-port="1"/>
		<edge from-layer="93" from-port="2" to-layer="95" to-port="0"/>
		<edge from-layer="94" from-port="1" to-layer="95" to-port="1"/>
		<edge from-layer="95" from-port="2" to-layer="97" to-port="0"/>
		<edge from-layer="96" from-port="1" to-layer="97" to-port="1"/>
		<edge from-layer="97" from-port="2" to-layer="98" to-port="0"/>
		<edge from-layer="98" from-port="1" to-layer="100" to-port="0"/>
		<edge from-layer="99" from-port="1" to-layer="100" to-port="1"/>
		<edge from-layer="100" from-port="2" to-layer="102" to-port="0"/>
		<edge from-layer="101" from-port="1" to-layer="102" to-port="1"/>
		<edge from-layer="102" from-port="2" to-layer="103" to-port="0"/>
		<edge from-layer="103" from-port="1" to-layer="105" to-port="0"/>
		<edge from-layer="104" from-port="1" to-layer="105" to-port="1"/>
		<edge from-layer="105" from-port="2" to-layer="107" to-port="0"/>
		<edge from-layer="106" from-port="1" to-layer="107" to-port="1"/>
		<edge from-layer="107" from-port="2" to-layer="109" to-port="0"/>
		<edge from-layer="108" from-port="1" to-layer="109" to-port="1"/>
		<edge from-layer="109" from-port="2" to-layer="111" to-port="0"/>
		<edge from-layer="110" from-port="1" to-layer="111" to-port="1"/>
		<edge from-layer="111" from-port="2" to-layer="112" to-port="0"/>
		<edge from-layer="112" from-port="1" to-layer="114" to-port="0"/>
		<edge from-layer="113" from-port="1" to-layer="114" to-port="1"/>
		<edge from-layer="114" from-port="2" to-layer="116" to-port="0"/>
		<edge from-layer="115" from-port="1" to-layer="116" to-port="1"/>
		<edge from-layer="116" from-port="2" to-layer="117" to-port="0"/>
		<edge from-layer="117" from-port="1" to-layer="119" to-port="0"/>
		<edge from-layer="118" from-port="1" to-layer="119" to-port="1"/>
		<edge from-layer="119" from-port="2" to-layer="121" to-port="0"/>
		<edge from-layer="120" from-port="1" to-layer="121" to-port="1"/>
		<edge from-layer="107" from-port="2" to-layer="122" to-port="0"/>
		<edge from-layer="121" from-port="2" to-layer="122" to-port="1"/>
		<edge from-layer="122" from-port="2" to-layer="124" to-port="0"/>
		<edge from-layer="123" from-port="1" to-layer="124" to-port="1"/>
		<edge from-layer="124" from-port="2" to-layer="126" to-port="0"/>
		<edge from-layer="125" from-port="1" to-layer="126" to-port="1"/>
		<edge from-layer="126" from-port="2" to-layer="127" to-port="0"/>
		<edge from-layer="127" from-port="1" to-layer="129" to-port="0"/>
		<edge from-layer="128" from-port="1" to-layer="129" to-port="1"/>
		<edge from-layer="129" from-port="2" to-layer="131" to-port="0"/>
		<edge from-layer="130" from-port="1" to-layer="131" to-port="1"/>
		<edge from-layer="131" from-port="2" to-layer="132" to-port="0"/>
		<edge from-layer="132" from-port="1" to-layer="134" to-port="0"/>
		<edge from-layer="133" from-port="1" to-layer="134" to-port="1"/>
		<edge from-layer="134" from-port="2" to-layer="136" to-port="0"/>
		<edge from-layer="135" from-port="1" to-layer="136" to-port="1"/>
		<edge from-layer="122" from-port="2" to-layer="137" to-port="0"/>
		<edge from-layer="136" from-port="2" to-layer="137" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="139" to-port="0"/>
		<edge from-layer="138" from-port="1" to-layer="139" to-port="1"/>
		<edge from-layer="139" from-port="2" to-layer="141" to-port="0"/>
		<edge from-layer="140" from-port="1" to-layer="141" to-port="1"/>
		<edge from-layer="141" from-port="2" to-layer="142" to-port="0"/>
		<edge from-layer="142" from-port="1" to-layer="144" to-port="0"/>
		<edge from-layer="143" from-port="1" to-layer="144" to-port="1"/>
		<edge from-layer="144" from-port="2" to-layer="146" to-port="0"/>
		<edge from-layer="145" from-port="1" to-layer="146" to-port="1"/>
		<edge from-layer="146" from-port="2" to-layer="147" to-port="0"/>
		<edge from-layer="147" from-port="1" to-layer="149" to-port="0"/>
		<edge from-layer="148" from-port="1" to-layer="149" to-port="1"/>
		<edge from-layer="149" from-port="2" to-layer="151" to-port="0"/>
		<edge from-layer="150" from-port="1" to-layer="151" to-port="1"/>
		<edge from-layer="137" from-port="2" to-layer="152" to-port="0"/>
		<edge from-layer="151" from-port="2" to-layer="152" to-port="1"/>
		<edge from-layer="152" from-port="2" to-layer="154" to-port="0"/>
		<edge from-layer="153" from-port="1" to-layer="154" to-port="1"/>
		<edge from-layer="154" from-port="2" to-layer="156" to-port="0"/>
		<edge from-layer="155" from-port="1" to-layer="156" to-port="1"/>
		<edge from-layer="156" from-port="2" to-layer="157" to-port="0"/>
		<edge from-layer="157" from-port="1" to-layer="159" to-port="0"/>
		<edge from-layer="158" from-port="1" to-layer="159" to-port="1"/>
		<edge from-layer="159" from-port="2" to-layer="161" to-port="0"/>
		<edge from-layer="160" from-port="1" to-layer="161" to-port="1"/>
		<edge from-layer="161" from-port="2" to-layer="162" to-port="0"/>
		<edge from-layer="162" from-port="1" to-layer="164" to-port="0"/>
		<edge from-layer="163" from-port="1" to-layer="164" to-port="1"/>
		<edge from-layer="164" from-port="2" to-layer="166" to-port="0"/>
		<edge from-layer="165" from-port="1" to-layer="166" to-port="1"/>
		<edge from-layer="166" from-port="2" to-layer="168" to-port="0"/>
		<edge from-layer="167" from-port="1" to-layer="168" to-port="1"/>
		<edge from-layer="168" from-port="2" to-layer="170" to-port="0"/>
		<edge from-layer="169" from-port="1" to-layer="170" to-port="1"/>
		<edge from-layer="170" from-port="2" to-layer="171" to-port="0"/>
		<edge from-layer="171" from-port="1" to-layer="173" to-port="0"/>
		<edge from-layer="172" from-port="1" to-layer="173" to-port="1"/>
		<edge from-layer="173" from-port="2" to-layer="175" to-port="0"/>
		<edge from-layer="174" from-port="1" to-layer="175" to-port="1"/>
		<edge from-layer="175" from-port="2" to-layer="176" to-port="0"/>
		<edge from-layer="176" from-port="1" to-layer="178" to-port="0"/>
		<edge from-layer="177" from-port="1" to-layer="178" to-port="1"/>
		<edge from-layer="178" from-port="2" to-layer="180" to-port="0"/>
		<edge from-layer="179" from-port="1" to-layer="180" to-port="1"/>
		<edge from-layer="166" from-port="2" to-layer="181" to-port="0"/>
		<edge from-layer="180" from-port="2" to-layer="181" to-port="1"/>
		<edge from-layer="181" from-port="2" to-layer="183" to-port="0"/>
		<edge from-layer="182" from-port="1" to-layer="183" to-port="1"/>
		<edge from-layer="183" from-port="2" to-layer="185" to-port="0"/>
		<edge from-layer="184" from-port="1" to-layer="185" to-port="1"/>
		<edge from-layer="185" from-port="2" to-layer="186" to-port="0"/>
		<edge from-layer="186" from-port="1" to-layer="188" to-port="0"/>
		<edge from-layer="187" from-port="1" to-layer="188" to-port="1"/>
		<edge from-layer="188" from-port="2" to-layer="190" to-port="0"/>
		<edge from-layer="189" from-port="1" to-layer="190" to-port="1"/>
		<edge from-layer="190" from-port="2" to-layer="191" to-port="0"/>
		<edge from-layer="191" from-port="1" to-layer="193" to-port="0"/>
		<edge from-layer="192" from-port="1" to-layer="193" to-port="1"/>
		<edge from-layer="193" from-port="2" to-layer="195" to-port="0"/>
		<edge from-layer="194" from-port="1" to-layer="195" to-port="1"/>
		<edge from-layer="181" from-port="2" to-layer="196" to-port="0"/>
		<edge from-layer="195" from-port="2" to-layer="196" to-port="1"/>
		<edge from-layer="196" from-port="2" to-layer="198" to-port="0"/>
		<edge from-layer="197" from-port="1" to-layer="198" to-port="1"/>
		<edge from-layer="198" from-port="2" to-layer="200" to-port="0"/>
		<edge from-layer="199" from-port="1" to-layer="200" to-port="1"/>
		<edge from-layer="200" from-port="2" to-layer="201" to-port="0"/>
		<edge from-layer="201" from-port="1" to-layer="203" to-port="0"/>
		<edge from-layer="202" from-port="1" to-layer="203" to-port="1"/>
		<edge from-layer="203" from-port="2" to-layer="205" to-port="0"/>
		<edge from-layer="204" from-port="1" to-layer="205" to-port="1"/>
		<edge from-layer="205" from-port="2" to-layer="206" to-port="0"/>
		<edge from-layer="206" from-port="1" to-layer="208" to-port="0"/>
		<edge from-layer="207" from-port="1" to-layer="208" to-port="1"/>
		<edge from-layer="208" from-port="2" to-layer="210" to-port="0"/>
		<edge from-layer="209" from-port="1" to-layer="210" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="212" to-port="0"/>
		<edge from-layer="211" from-port="1" to-layer="212" to-port="1"/>
		<edge from-layer="210" from-port="2" to-layer="213" to-port="0"/>
		<edge from-layer="213" from-port="1" to-layer="216" to-port="0"/>
		<edge from-layer="214" from-port="1" to-layer="216" to-port="1"/>
		<edge from-layer="215" from-port="1" to-layer="216" to-port="2"/>
		<edge from-layer="216" from-port="3" to-layer="218" to-port="0"/>
		<edge from-layer="217" from-port="1" to-layer="218" to-port="1"/>
		<edge from-layer="218" from-port="2" to-layer="220" to-port="0"/>
		<edge from-layer="219" from-port="1" to-layer="220" to-port="1"/>
		<edge from-layer="212" from-port="2" to-layer="221" to-port="0"/>
		<edge from-layer="220" from-port="2" to-layer="221" to-port="1"/>
		<edge from-layer="201" from-port="1" to-layer="223" to-port="0"/>
		<edge from-layer="222" from-port="1" to-layer="223" to-port="1"/>
		<edge from-layer="223" from-port="2" to-layer="225" to-port="0"/>
		<edge from-layer="224" from-port="1" to-layer="225" to-port="1"/>
		<edge from-layer="225" from-port="2" to-layer="226" to-port="0"/>
		<edge from-layer="226" from-port="1" to-layer="228" to-port="0"/>
		<edge from-layer="227" from-port="1" to-layer="228" to-port="1"/>
		<edge from-layer="228" from-port="2" to-layer="230" to-port="0"/>
		<edge from-layer="229" from-port="1" to-layer="230" to-port="1"/>
		<edge from-layer="230" from-port="2" to-layer="232" to-port="0"/>
		<edge from-layer="231" from-port="1" to-layer="232" to-port="1"/>
		<edge from-layer="232" from-port="2" to-layer="234" to-port="0"/>
		<edge from-layer="233" from-port="1" to-layer="234" to-port="1"/>
		<edge from-layer="234" from-port="2" to-layer="235" to-port="0"/>
		<edge from-layer="235" from-port="1" to-layer="237" to-port="0"/>
		<edge from-layer="236" from-port="1" to-layer="237" to-port="1"/>
		<edge from-layer="237" from-port="2" to-layer="239" to-port="0"/>
		<edge from-layer="238" from-port="1" to-layer="239" to-port="1"/>
		<edge from-layer="239" from-port="2" to-layer="240" to-port="0"/>
		<edge from-layer="240" from-port="1" to-layer="242" to-port="0"/>
		<edge from-layer="241" from-port="1" to-layer="242" to-port="1"/>
		<edge from-layer="242" from-port="2" to-layer="244" to-port="0"/>
		<edge from-layer="243" from-port="1" to-layer="244" to-port="1"/>
		<edge from-layer="230" from-port="2" to-layer="245" to-port="0"/>
		<edge from-layer="244" from-port="2" to-layer="245" to-port="1"/>
		<edge from-layer="245" from-port="2" to-layer="247" to-port="0"/>
		<edge from-layer="246" from-port="1" to-layer="247" to-port="1"/>
		<edge from-layer="247" from-port="2" to-layer="249" to-port="0"/>
		<edge from-layer="248" from-port="1" to-layer="249" to-port="1"/>
		<edge from-layer="249" from-port="2" to-layer="250" to-port="0"/>
		<edge from-layer="250" from-port="1" to-layer="252" to-port="0"/>
		<edge from-layer="251" from-port="1" to-layer="252" to-port="1"/>
		<edge from-layer="252" from-port="2" to-layer="254" to-port="0"/>
		<edge from-layer="253" from-port="1" to-layer="254" to-port="1"/>
		<edge from-layer="254" from-port="2" to-layer="255" to-port="0"/>
		<edge from-layer="255" from-port="1" to-layer="257" to-port="0"/>
		<edge from-layer="256" from-port="1" to-layer="257" to-port="1"/>
		<edge from-layer="257" from-port="2" to-layer="259" to-port="0"/>
		<edge from-layer="258" from-port="1" to-layer="259" to-port="1"/>
		<edge from-layer="245" from-port="2" to-layer="260" to-port="0"/>
		<edge from-layer="259" from-port="2" to-layer="260" to-port="1"/>
		<edge from-layer="260" from-port="2" to-layer="262" to-port="0"/>
		<edge from-layer="261" from-port="1" to-layer="262" to-port="1"/>
		<edge from-layer="262" from-port="2" to-layer="264" to-port="0"/>
		<edge from-layer="263" from-port="1" to-layer="264" to-port="1"/>
		<edge from-layer="264" from-port="2" to-layer="265" to-port="0"/>
		<edge from-layer="265" from-port="1" to-layer="267" to-port="0"/>
		<edge from-layer="266" from-port="1" to-layer="267" to-port="1"/>
		<edge from-layer="267" from-port="2" to-layer="269" to-port="0"/>
		<edge from-layer="268" from-port="1" to-layer="269" to-port="1"/>
		<edge from-layer="269" from-port="2" to-layer="270" to-port="0"/>
		<edge from-layer="270" from-port="1" to-layer="272" to-port="0"/>
		<edge from-layer="271" from-port="1" to-layer="272" to-port="1"/>
		<edge from-layer="272" from-port="2" to-layer="274" to-port="0"/>
		<edge from-layer="273" from-port="1" to-layer="274" to-port="1"/>
		<edge from-layer="274" from-port="2" to-layer="276" to-port="0"/>
		<edge from-layer="275" from-port="1" to-layer="276" to-port="1"/>
		<edge from-layer="276" from-port="2" to-layer="278" to-port="0"/>
		<edge from-layer="277" from-port="1" to-layer="278" to-port="1"/>
		<edge from-layer="278" from-port="2" to-layer="279" to-port="0"/>
		<edge from-layer="279" from-port="1" to-layer="281" to-port="0"/>
		<edge from-layer="280" from-port="1" to-layer="281" to-port="1"/>
		<edge from-layer="281" from-port="2" to-layer="283" to-port="0"/>
		<edge from-layer="282" from-port="1" to-layer="283" to-port="1"/>
		<edge from-layer="283" from-port="2" to-layer="284" to-port="0"/>
		<edge from-layer="284" from-port="1" to-layer="286" to-port="0"/>
		<edge from-layer="285" from-port="1" to-layer="286" to-port="1"/>
		<edge from-layer="286" from-port="2" to-layer="288" to-port="0"/>
		<edge from-layer="287" from-port="1" to-layer="288" to-port="1"/>
		<edge from-layer="288" from-port="2" to-layer="290" to-port="0"/>
		<edge from-layer="289" from-port="1" to-layer="290" to-port="1"/>
		<edge from-layer="288" from-port="2" to-layer="291" to-port="0"/>
		<edge from-layer="291" from-port="1" to-layer="294" to-port="0"/>
		<edge from-layer="292" from-port="1" to-layer="294" to-port="1"/>
		<edge from-layer="293" from-port="1" to-layer="294" to-port="2"/>
		<edge from-layer="294" from-port="3" to-layer="296" to-port="0"/>
		<edge from-layer="295" from-port="1" to-layer="296" to-port="1"/>
		<edge from-layer="296" from-port="2" to-layer="298" to-port="0"/>
		<edge from-layer="297" from-port="1" to-layer="298" to-port="1"/>
		<edge from-layer="290" from-port="2" to-layer="299" to-port="0"/>
		<edge from-layer="298" from-port="2" to-layer="299" to-port="1"/>
		<edge from-layer="279" from-port="1" to-layer="301" to-port="0"/>
		<edge from-layer="300" from-port="1" to-layer="301" to-port="1"/>
		<edge from-layer="301" from-port="2" to-layer="303" to-port="0"/>
		<edge from-layer="302" from-port="1" to-layer="303" to-port="1"/>
		<edge from-layer="303" from-port="2" to-layer="304" to-port="0"/>
		<edge from-layer="304" from-port="1" to-layer="306" to-port="0"/>
		<edge from-layer="305" from-port="1" to-layer="306" to-port="1"/>
		<edge from-layer="306" from-port="2" to-layer="308" to-port="0"/>
		<edge from-layer="307" from-port="1" to-layer="308" to-port="1"/>
		<edge from-layer="308" from-port="2" to-layer="309" to-port="0"/>
		<edge from-layer="309" from-port="1" to-layer="311" to-port="0"/>
		<edge from-layer="310" from-port="1" to-layer="311" to-port="1"/>
		<edge from-layer="311" from-port="2" to-layer="313" to-port="0"/>
		<edge from-layer="312" from-port="1" to-layer="313" to-port="1"/>
		<edge from-layer="313" from-port="2" to-layer="314" to-port="0"/>
		<edge from-layer="314" from-port="1" to-layer="316" to-port="0"/>
		<edge from-layer="315" from-port="1" to-layer="316" to-port="1"/>
		<edge from-layer="316" from-port="2" to-layer="318" to-port="0"/>
		<edge from-layer="317" from-port="1" to-layer="318" to-port="1"/>
		<edge from-layer="318" from-port="2" to-layer="319" to-port="0"/>
		<edge from-layer="319" from-port="1" to-layer="321" to-port="0"/>
		<edge from-layer="320" from-port="1" to-layer="321" to-port="1"/>
		<edge from-layer="321" from-port="2" to-layer="323" to-port="0"/>
		<edge from-layer="322" from-port="1" to-layer="323" to-port="1"/>
		<edge from-layer="323" from-port="2" to-layer="325" to-port="0"/>
		<edge from-layer="324" from-port="1" to-layer="325" to-port="1"/>
		<edge from-layer="323" from-port="2" to-layer="326" to-port="0"/>
		<edge from-layer="326" from-port="1" to-layer="329" to-port="0"/>
		<edge from-layer="327" from-port="1" to-layer="329" to-port="1"/>
		<edge from-layer="328" from-port="1" to-layer="329" to-port="2"/>
		<edge from-layer="329" from-port="3" to-layer="331" to-port="0"/>
		<edge from-layer="330" from-port="1" to-layer="331" to-port="1"/>
		<edge from-layer="331" from-port="2" to-layer="333" to-port="0"/>
		<edge from-layer="332" from-port="1" to-layer="333" to-port="1"/>
		<edge from-layer="325" from-port="2" to-layer="334" to-port="0"/>
		<edge from-layer="333" from-port="2" to-layer="334" to-port="1"/>
		<edge from-layer="221" from-port="2" to-layer="335" to-port="0"/>
		<edge from-layer="299" from-port="2" to-layer="335" to-port="1"/>
		<edge from-layer="334" from-port="2" to-layer="335" to-port="2"/>
		<edge from-layer="201" from-port="1" to-layer="337" to-port="0"/>
		<edge from-layer="336" from-port="1" to-layer="337" to-port="1"/>
		<edge from-layer="337" from-port="2" to-layer="339" to-port="0"/>
		<edge from-layer="338" from-port="1" to-layer="339" to-port="1"/>
		<edge from-layer="339" from-port="2" to-layer="340" to-port="0"/>
		<edge from-layer="340" from-port="1" to-layer="342" to-port="0"/>
		<edge from-layer="341" from-port="1" to-layer="342" to-port="1"/>
		<edge from-layer="342" from-port="2" to-layer="344" to-port="0"/>
		<edge from-layer="343" from-port="1" to-layer="344" to-port="1"/>
		<edge from-layer="344" from-port="2" to-layer="346" to-port="0"/>
		<edge from-layer="345" from-port="1" to-layer="346" to-port="1"/>
		<edge from-layer="344" from-port="2" to-layer="347" to-port="0"/>
		<edge from-layer="347" from-port="1" to-layer="350" to-port="0"/>
		<edge from-layer="348" from-port="1" to-layer="350" to-port="1"/>
		<edge from-layer="349" from-port="1" to-layer="350" to-port="2"/>
		<edge from-layer="350" from-port="3" to-layer="352" to-port="0"/>
		<edge from-layer="351" from-port="1" to-layer="352" to-port="1"/>
		<edge from-layer="352" from-port="2" to-layer="354" to-port="0"/>
		<edge from-layer="353" from-port="1" to-layer="354" to-port="1"/>
		<edge from-layer="346" from-port="2" to-layer="355" to-port="0"/>
		<edge from-layer="354" from-port="2" to-layer="355" to-port="1"/>
		<edge from-layer="279" from-port="1" to-layer="357" to-port="0"/>
		<edge from-layer="356" from-port="1" to-layer="357" to-port="1"/>
		<edge from-layer="357" from-port="2" to-layer="359" to-port="0"/>
		<edge from-layer="358" from-port="1" to-layer="359" to-port="1"/>
		<edge from-layer="359" from-port="2" to-layer="360" to-port="0"/>
		<edge from-layer="360" from-port="1" to-layer="362" to-port="0"/>
		<edge from-layer="361" from-port="1" to-layer="362" to-port="1"/>
		<edge from-layer="362" from-port="2" to-layer="364" to-port="0"/>
		<edge from-layer="363" from-port="1" to-layer="364" to-port="1"/>
		<edge from-layer="364" from-port="2" to-layer="366" to-port="0"/>
		<edge from-layer="365" from-port="1" to-layer="366" to-port="1"/>
		<edge from-layer="364" from-port="2" to-layer="367" to-port="0"/>
		<edge from-layer="367" from-port="1" to-layer="370" to-port="0"/>
		<edge from-layer="368" from-port="1" to-layer="370" to-port="1"/>
		<edge from-layer="369" from-port="1" to-layer="370" to-port="2"/>
		<edge from-layer="370" from-port="3" to-layer="372" to-port="0"/>
		<edge from-layer="371" from-port="1" to-layer="372" to-port="1"/>
		<edge from-layer="372" from-port="2" to-layer="374" to-port="0"/>
		<edge from-layer="373" from-port="1" to-layer="374" to-port="1"/>
		<edge from-layer="366" from-port="2" to-layer="375" to-port="0"/>
		<edge from-layer="374" from-port="2" to-layer="375" to-port="1"/>
		<edge from-layer="314" from-port="1" to-layer="377" to-port="0"/>
		<edge from-layer="376" from-port="1" to-layer="377" to-port="1"/>
		<edge from-layer="377" from-port="2" to-layer="379" to-port="0"/>
		<edge from-layer="378" from-port="1" to-layer="379" to-port="1"/>
		<edge from-layer="379" from-port="2" to-layer="380" to-port="0"/>
		<edge from-layer="380" from-port="1" to-layer="382" to-port="0"/>
		<edge from-layer="381" from-port="1" to-layer="382" to-port="1"/>
		<edge from-layer="382" from-port="2" to-layer="384" to-port="0"/>
		<edge from-layer="383" from-port="1" to-layer="384" to-port="1"/>
		<edge from-layer="384" from-port="2" to-layer="386" to-port="0"/>
		<edge from-layer="385" from-port="1" to-layer="386" to-port="1"/>
		<edge from-layer="384" from-port="2" to-layer="387" to-port="0"/>
		<edge from-layer="387" from-port="1" to-layer="390" to-port="0"/>
		<edge from-layer="388" from-port="1" to-layer="390" to-port="1"/>
		<edge from-layer="389" from-port="1" to-layer="390" to-port="2"/>
		<edge from-layer="390" from-port="3" to-layer="392" to-port="0"/>
		<edge from-layer="391" from-port="1" to-layer="392" to-port="1"/>
		<edge from-layer="392" from-port="2" to-layer="394" to-port="0"/>
		<edge from-layer="393" from-port="1" to-layer="394" to-port="1"/>
		<edge from-layer="386" from-port="2" to-layer="395" to-port="0"/>
		<edge from-layer="394" from-port="2" to-layer="395" to-port="1"/>
		<edge from-layer="355" from-port="2" to-layer="396" to-port="0"/>
		<edge from-layer="375" from-port="2" to-layer="396" to-port="1"/>
		<edge from-layer="395" from-port="2" to-layer="396" to-port="2"/>
		<edge from-layer="396" from-port="3" to-layer="397" to-port="0"/>
		<edge from-layer="397" from-port="1" to-layer="400" to-port="0"/>
		<edge from-layer="398" from-port="1" to-layer="400" to-port="1"/>
		<edge from-layer="399" from-port="1" to-layer="400" to-port="2"/>
		<edge from-layer="400" from-port="3" to-layer="402" to-port="0"/>
		<edge from-layer="401" from-port="1" to-layer="402" to-port="1"/>
		<edge from-layer="402" from-port="2" to-layer="405" to-port="0"/>
		<edge from-layer="403" from-port="1" to-layer="405" to-port="1"/>
		<edge from-layer="404" from-port="1" to-layer="405" to-port="2"/>
		<edge from-layer="396" from-port="3" to-layer="406" to-port="0"/>
		<edge from-layer="405" from-port="3" to-layer="406" to-port="1"/>
		<edge from-layer="406" from-port="2" to-layer="407" to-port="0"/>
		<edge from-layer="407" from-port="1" to-layer="410" to-port="0"/>
		<edge from-layer="408" from-port="1" to-layer="410" to-port="1"/>
		<edge from-layer="409" from-port="1" to-layer="410" to-port="2"/>
		<edge from-layer="410" from-port="3" to-layer="412" to-port="0"/>
		<edge from-layer="411" from-port="1" to-layer="412" to-port="1"/>
		<edge from-layer="412" from-port="2" to-layer="414" to-port="0"/>
		<edge from-layer="413" from-port="1" to-layer="414" to-port="1"/>
		<edge from-layer="406" from-port="2" to-layer="415" to-port="0"/>
		<edge from-layer="414" from-port="2" to-layer="415" to-port="1"/>
		<edge from-layer="415" from-port="2" to-layer="416" to-port="0"/>
		<edge from-layer="406" from-port="2" to-layer="417" to-port="0"/>
		<edge from-layer="416" from-port="1" to-layer="418" to-port="0"/>
		<edge from-layer="417" from-port="1" to-layer="418" to-port="1"/>
		<edge from-layer="418" from-port="2" to-layer="419" to-port="0"/>
		<edge from-layer="419" from-port="1" to-layer="422" to-port="0"/>
		<edge from-layer="420" from-port="1" to-layer="422" to-port="1"/>
		<edge from-layer="421" from-port="1" to-layer="422" to-port="2"/>
		<edge from-layer="422" from-port="3" to-layer="424" to-port="0"/>
		<edge from-layer="423" from-port="1" to-layer="424" to-port="1"/>
		<edge from-layer="424" from-port="2" to-layer="426" to-port="0"/>
		<edge from-layer="425" from-port="1" to-layer="426" to-port="1"/>
		<edge from-layer="418" from-port="2" to-layer="427" to-port="0"/>
		<edge from-layer="426" from-port="2" to-layer="427" to-port="1"/>
		<edge from-layer="201" from-port="1" to-layer="428" to-port="0"/>
		<edge from-layer="428" from-port="1" to-layer="432" to-port="0"/>
		<edge from-layer="429" from-port="1" to-layer="432" to-port="1"/>
		<edge from-layer="430" from-port="1" to-layer="432" to-port="2"/>
		<edge from-layer="431" from-port="1" to-layer="432" to-port="3"/>
		<edge from-layer="2" from-port="2" to-layer="433" to-port="0"/>
		<edge from-layer="433" from-port="1" to-layer="437" to-port="0"/>
		<edge from-layer="434" from-port="1" to-layer="437" to-port="1"/>
		<edge from-layer="435" from-port="1" to-layer="437" to-port="2"/>
		<edge from-layer="436" from-port="1" to-layer="437" to-port="3"/>
		<edge from-layer="432" from-port="4" to-layer="438" to-port="0"/>
		<edge from-layer="437" from-port="4" to-layer="438" to-port="1"/>
		<edge from-layer="438" from-port="2" to-layer="440" to-port="0"/>
		<edge from-layer="439" from-port="1" to-layer="440" to-port="1"/>
		<edge from-layer="279" from-port="1" to-layer="441" to-port="0"/>
		<edge from-layer="441" from-port="1" to-layer="445" to-port="0"/>
		<edge from-layer="442" from-port="1" to-layer="445" to-port="1"/>
		<edge from-layer="443" from-port="1" to-layer="445" to-port="2"/>
		<edge from-layer="444" from-port="1" to-layer="445" to-port="3"/>
		<edge from-layer="2" from-port="2" to-layer="446" to-port="0"/>
		<edge from-layer="446" from-port="1" to-layer="450" to-port="0"/>
		<edge from-layer="447" from-port="1" to-layer="450" to-port="1"/>
		<edge from-layer="448" from-port="1" to-layer="450" to-port="2"/>
		<edge from-layer="449" from-port="1" to-layer="450" to-port="3"/>
		<edge from-layer="445" from-port="4" to-layer="451" to-port="0"/>
		<edge from-layer="450" from-port="4" to-layer="451" to-port="1"/>
		<edge from-layer="451" from-port="2" to-layer="453" to-port="0"/>
		<edge from-layer="452" from-port="1" to-layer="453" to-port="1"/>
		<edge from-layer="314" from-port="1" to-layer="454" to-port="0"/>
		<edge from-layer="454" from-port="1" to-layer="458" to-port="0"/>
		<edge from-layer="455" from-port="1" to-layer="458" to-port="1"/>
		<edge from-layer="456" from-port="1" to-layer="458" to-port="2"/>
		<edge from-layer="457" from-port="1" to-layer="458" to-port="3"/>
		<edge from-layer="2" from-port="2" to-layer="459" to-port="0"/>
		<edge from-layer="459" from-port="1" to-layer="463" to-port="0"/>
		<edge from-layer="460" from-port="1" to-layer="463" to-port="1"/>
		<edge from-layer="461" from-port="1" to-layer="463" to-port="2"/>
		<edge from-layer="462" from-port="1" to-layer="463" to-port="3"/>
		<edge from-layer="458" from-port="4" to-layer="464" to-port="0"/>
		<edge from-layer="463" from-port="4" to-layer="464" to-port="1"/>
		<edge from-layer="464" from-port="2" to-layer="466" to-port="0"/>
		<edge from-layer="465" from-port="1" to-layer="466" to-port="1"/>
		<edge from-layer="440" from-port="2" to-layer="467" to-port="0"/>
		<edge from-layer="453" from-port="2" to-layer="467" to-port="1"/>
		<edge from-layer="466" from-port="2" to-layer="467" to-port="2"/>
		<edge from-layer="335" from-port="3" to-layer="468" to-port="0"/>
		<edge from-layer="427" from-port="2" to-layer="468" to-port="1"/>
		<edge from-layer="467" from-port="3" to-layer="468" to-port="2"/>
		<edge from-layer="468" from-port="3" to-layer="469" to-port="0"/>
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
			<input value="input.1"/>
			<input_model value="DIR/person-vehicle-bike-detection-crossroad-1016.onnx"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1,3,512,512]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'input.1': {'mean': None, 'scale': array([255.])}}"/>
			<mean_values value="()"/>
			<model_name value="person-vehicle-bike-detection-crossroad-1016"/>
			<output_dir value="DIR"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'input.1': array([  1,   3, 512, 512])}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input.1[255.0]"/>
			<silent value="False"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, output, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
