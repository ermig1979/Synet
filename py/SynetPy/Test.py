###################################################################################################
# Synet Framework (http://github.com/ermig1979/Synet).
#
# Copyright (c) 2018-2024 Yermalayeu Ihar.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###################################################################################################

import argparse
import os
import ctypes
import pathlib
import sys

import Synet
import Simd

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Synet", description="Synet Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default="")
	parser.add_argument("-m", "--model", help="Path to model file.", required=False, type=str, default="./data/precision/detection/test_003/sy_fp32_v0.xml")
	parser.add_argument("-w", "--weight", help="Path to weight file.", required=False, type=str, default="./data/precision/detection/test_003/synet.bin")
	parser.add_argument("-i", "--image", help="Path to image file.", required=False, type=str, default="./data/images/faces/faces_000.jpg")
	args = parser.parse_args()

	Simd.Lib.Init(args.bin)
	Synet.Lib.Init(args.bin)
	
	print("Synet: {0}; {1} \n".format(Synet.Lib.Version(), Simd.Lib.SysInfo()))
	
	network = Synet.Network()
	
	print("Load network from {0} and {1}: ".format(args.model, args.weight), end="")
	if not network.Load(args.model, args.weight) :
		print("Error. Can't load!")
		return 1
	print("OK.")

	print("Set batch = 1.")	
	network.SetBatch(1)
	
	print("Network has {0} inputs:".format(network.SrcSize()))
	for src in range(network.SrcSize()) :
		tensor = network.Src(src)
		print(" {0}: {1} {2} {3} {4}".format(src, tensor.Name(), tensor.Shape(), tensor.Type().name, tensor.Format().name))
		
	print("Network has {0} outputs:".format(network.DstSize()))
	for dst in range(network.DstSize()) :
		tensor = network.Dst(dst)
		print(" {0}: {1} {2} {3} {4}".format(dst, tensor.Name(), tensor.Shape(), tensor.Type().name, tensor.Format().name))

	print("\nLoad test image {0} :".format(args.image), end="")	
	image = Simd.Image()
	if not image.Load(args.image) :
		print("Error. Can't load!")
		return 1
	print("OK.")
	
	print("Set network input: ", end = "")	
	shape = network.NchwShape()
	resized = Simd.Resized(image, shape[3], shape[2], Simd.ResizeMethod.Area)
	lower = [0.0, 0.0, 0.0]
	upper = [255.0, 255.0, 255.0]
	input = network.Src(0)
	Simd.SynetSetInput(resized, lower, upper, input.Data(), shape[1], input.Format())
	print("OK.")	
	
	network.Forward()
	print("Network inference.")		
	
	dst = network.Dst(0)
	shp = dst.Shape()
	print("Network output {0} :".format(shp))	
	ptr = dst.As32f()
	for i in range(shp[2]) :
		if ptr[i * 7 + 2] < 0.5 :
			break
		print(" {0} : {1} {2} {3} {4} {5} {6} {7} ".format(i, ptr[i*7 + 0], ptr[i*7 + 1], ptr[i*7 + 2], ptr[i*7 + 3], ptr[i*7 + 4], ptr[i*7 + 5], ptr[i*7 + 6]))
	
	print("\nSynet Python Wrapper ended successfully!\n")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
