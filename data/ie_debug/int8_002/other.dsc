<?xml version="1.0"?>
<net name="test" version="3" batch="1">
	<layers>
		<layer name="data" type="Input" precision="FP32" id="0">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>18</dim>
				</port>
			</output>
		</layer>
		<layer name="Scale1" type="ScaleShift" precision="FP32" id="1">
			<data broadcast="0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>18</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>18</dim>
				</port>
			</output>
			<blobs>
				<biases offset="4" size="4" />
				<weights offset="0" size="4" />
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
	</edges>
	<statistics>
		<layer>
			<name>Scale1</name>
			<min>-255.0</min>
			<max>255.0</max>
		</layer>
		<layer>
			<name>data</name>
			<min>0.0</min>
			<max>255.0</max>
		</layer>
	</statistics>
</net>
