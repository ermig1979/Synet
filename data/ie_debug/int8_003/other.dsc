<?xml version="1.0"?>
<net name="test" version="3" batch="1">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer name="Conv1" type="Convolution" precision="FP32" id="1">
			<data dilations="1,1" group="3" kernel="1,1" output="3" pads_begin="0,0" pads_end="0,0" quantization_level="I8" strides="1,1" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<biases offset="12" size="12" />
				<weights offset="0" size="12" />
			</blobs>
		</layer>
		<layer name="Relu1" type="ReLU" precision="FP32" id="2">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer name="Conv2" type="Convolution" precision="FP32" id="3">
			<data dilations="1,1" group="3" kernel="1,1" output="3" pads_begin="0,0" pads_end="0,0" quantization_level="I8" strides="1,1" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<biases offset="36" size="12" />
				<weights offset="24" size="12" />
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
	</edges>
	<statistics>
		<layer>
			<name>Conv1</name>
			<min>-255.0, -255.0, -255.0</min>
			<max>255.0, 255.0, 255.0</max>
		</layer>
		<layer>
			<name>Conv2</name>
			<min>-355.0, -355.0, -355.0</min>
			<max>355.0, 355.0, 355.0</max>
		</layer>
		<layer>
			<name>data</name>
			<min>0.0, 0.0, 0.0</min>
			<max>255.0, 255.0, 255.0</max>
		</layer>
		<layer>
			<name>Relu1</name>
			<min>0.0, 0.0, 0.0</min>
			<max>255.0, 255.0, 255.0</max>
		</layer>
	</statistics>
</net>
