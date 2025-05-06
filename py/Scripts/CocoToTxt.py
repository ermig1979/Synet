###################################################################################################
# Synet Framework (http://github.com/ermig1979/Synet).
#
# Copyright (c) 2018-2023 Yermalayeu Ihar.
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
import json

###################################################################################################

def Error(text = "") :
	if text != "" :
		print(text)
	return 1

###################################################################################################

class Object :
	def __init__(self):
		self.x = 0
		self.y = 0
		self.w = 0
		self.h = 0
		self.id = 0

###################################################################################################

class Image :
	def __init__(self):
		self.name = ""
		self.objects = []

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="CocoToTxt", description="COCO to TXT converter.")
	parser.add_argument("-s", "--src", help="Path to input COCO(JSON) file.", required=False, type=str, default="src.json")
	parser.add_argument("-d", "--dst", help="Path to output TXT file.", required=False, type=str, default="dst.txt")
	parser.add_argument("-a", "--all", help="Path to output file list.", required=False, type=str, default="all.txt")
	args = parser.parse_args()
	
	if not os.path.isfile(args.src):
		return Error("Input file '{0}' is not exist!".format(args.src))
	
	src = open(args.src, "r")
	data = json.load(src)
	
	images = []
	for img in data["images"]:
		image = Image()
		image.name = img["file_name"]
		images.append(image)

	for ann in data["annotations"]:
		obj = Object()
		obj.id = ann["category_id"]
		obj.x = ann["bbox"][0]
		obj.y = ann["bbox"][1]
		obj.w = ann["bbox"][2]
		obj.h = ann["bbox"][3]
		images[ann["image_id"]].objects.append(obj)
	
	src.close()
	
	dst = open(args.dst, "w")
	for img in images:
		dst.write("{0}\n".format(img.name))
		dst.write("{0}\n".format(len(img.objects)))
		o = 0
		for obj in img.objects:
			dst.write("{0} {1} {2} {3} {4} 0 0 0 0 {5}\n".format(obj.x, obj.y, obj.w, obj.h, obj.id, o))
			o += 1
	dst.close()
	
	all = open(args.all, "w")
	for img in images:
		all.write("{0}\n".format(img.name))
	all.close()
	
	print("Conversion is successfull!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
