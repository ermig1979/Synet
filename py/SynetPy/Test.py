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

def DrawRectangle(image : Simd.Image, left : int, top : int, right : int, bottom : int, color: int) :
	if image.Format() != Simd.PixelFormat.Bgra32 :
		raise Exception("DrawRectangle can annotate only for BGRA-32 image!")
	bgra = ctypes.c_uint32(color)
	data = ctypes.cast(image.Data(), ctypes.POINTER(ctypes.c_uint32))
	stride = image.Stride() // 4
	if top >= 0 and top < image.Height() :
		beg = top * stride + max(0, left)
		end = top * stride + min(right, image.Width())
		for offs in range(beg, end) :
			data[offs] = bgra
	if bottom > 0 and bottom <= image.Height() :
		beg = (bottom - 1) * stride + max(0, left)
		end = (bottom - 1) * stride + min(right, image.Width())
		for offs in range(beg, end) :
			data[offs] = bgra
	if left >= 0 and left < image.Width() :
		beg = max(0, top) * stride + left
		end = min(bottom, image.Height()) * stride + left
		for offs in range(beg, end, stride) :
			data[offs] = bgra
	if right > 0 and right < image.Width() :
		beg = max(0, top) * stride + right - 1
		end = min(bottom, image.Height()) * stride + right - 1
		for offs in range(beg, end, stride) :
			data[offs] = bgra
	
###################################################################################################

def AnnotateDetection(image : Simd.Image, ptr : ctypes.POINTER(ctypes.c_float), offs : int) :
	color = int("ff00ff00", 16)
	left = int(ptr[offs + 3] * image.Width())
	top = int(ptr[offs + 4] * image.Height())
	right = int(ptr[offs + 5] * image.Width())
	bottom = int(ptr[offs + 6] * image.Height())
	DrawRectangle(image, left, top, right, bottom, color)

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Synet", description="Synet Python Wrapper Tests.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default="")
	parser.add_argument("-m", "--model", help="Path to model file.", required=False, type=str, default="./data/precision/detection/test_003/sy_fp32_v0.xml")
	parser.add_argument("-w", "--weight", help="Path to weight file.", required=False, type=str, default="./data/precision/detection/test_003/synet.bin")
	parser.add_argument("-i", "--image", help="Path to image file.", required=False, type=str, default="./data/images/faces/faces_000.jpg")
	parser.add_argument("-o", "--output", help="Output directory.", required=False, type=str, default=pathlib.Path(__file__).parent.resolve())
	args = parser.parse_args()

	Simd.Lib.Init(args.bin)
	Synet.Lib.Init(args.bin)
	
	Synet.Lib.SetConsoleLogLevel(Synet.LogLevel.Info)
	
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
	
	annotated = image.Converted(Simd.PixelFormat.Bgra32)
	dst = network.Dst(0)
	shp = dst.Shape()
	ptr = dst.As32f()
	faces  = 0;
	for i in range(shp[2]) :
		if ptr[i * 7 + 2] < 0.5 :
			break
		faces += 1
		AnnotateDetection(annotated, ptr, i * 7)
	outPath = "{0}/annotated_faces.jpg".format(args.output)
	print("There are {0} faces were found. See image {1}.".format(faces, outPath))	
	annotated.Save(outPath)	

	print("\nSynet Python Wrapper Tests ended successfully!\n")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
