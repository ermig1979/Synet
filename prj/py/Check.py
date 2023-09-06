import argparse
from http.server import ThreadingHTTPServer
import os
from datetime import datetime

###################################################################################################

class Test():
	def __init__(self):
		self.framework = ""
		self.group = ""
		self.name = ""
		self.image = ""
		self.batch = False
		self.path = ""
		
###################################################################################################

def GetTestList(args):
	tests = []
	listPath = args.src + os.path.sep + "tests.txt"
	print(listPath)
	if not os.path.isfile(listPath):
		print("Test list '{}' is not exist!".format(listPath))
		return False, tests
	listFile = open(listPath, "r")
	while True:
		line = listFile.readline()
		if not line:
			break
		vals = line.split()
		if len(vals) < 5 :
			continue
		test = Test()
		test.framework = vals[0]
		test.group = vals[1]
		test.name = vals[2]
		test.image = vals[3]
		test.batch = vals[4]
		
		if not (test.framework == "onnx" or test.framework == "inference_engine") :
			return False, tests
		
		test.path = test.framework
		if test.group != "root" :
			test.path += os.path.sep + test.group
		test.path += os.path.sep + test.name
		if args.include != "" and test.path.find(args.include) == -1:
			continue
		if args.exclude != "" and test.path.find(args.exclude) != -1:
			continue
		
		tests.append(test)
	listFile.close()
	return True, tests

###################################################################################################

def RunTest(args, dstPath, test, format, batch):
	binPath = args.bin + os.path.sep + "test_" + test.framework
	if not os.path.isfile(binPath):
		print("Binary '{}' is not exist!".format(binPath))
		return False
	
	testPath = args.src + os.path.sep + test.path
	if not os.path.isdir(testPath):
		print("Test directory '{}' is not exist!".format(testPath))
		return False
	
	imagePath = ""
	if test.image == "local" :
		imagePath = testPath + os.path.sep + "image"
	else :
		imagePath = args.src + os.path.sep + "images" + os.path.sep + test.image
	if not os.path.isdir(imagePath):
		print("Image directory '{}' is not exist!".format(imagePath))
		return False
	
	logPath = ""
	if test.group == "root" :
		logPath = dstPath + os.path.sep + "c{0}_{1}_f{2}_b{3}.txt".format(test.framework[0], test.name, format, batch)
	else :
		logPath = dstPath + os.path.sep + "c{0}_{1}__{2}_f{3}_b{4}.txt".format(test.framework[0], test.group, test.name, format, batch)

	threshold = 0.0031
	pathArgs = "-fm={0}/other.dsc -fw={0}/other.dat -sm={0}/synet{1}.xml -sw={0}/synet{1}.bin -id={2} -od={0}/output -tp={0}/param.xml".format(testPath, format, imagePath)
	
	print("Test . log is {0}".format(logPath))
	
	trashFile = imagePath + os.path.sep + "descript.ion"
	if os.path.isfile(trashFile) :
		os.remove(trashFile)
		
	if not args.fast and batch == 1 :
		cmd = "{0} -m=convert {1} -tf={2} -cs=1".format(binPath, pathArgs, format)
		result = os.system(cmd)
		if result != 0:
			return False
		
	num = 10 if args.fast else 2
	cmd = "{0} -m=compare -e=3 {1} -rn=1 -wt=1 -tt=0 -ie={2} -be={2} -tf={3} -bs={4} -ct={5} -cs=1 -ln={6}".format(binPath, pathArgs, num, format, batch, threshold, logPath)
	result = os.system(cmd)
	if result != 0:
		return False
	
	return True

###################################################################################################

def RunTests(args, dstPath, test):
	if not RunTest(args, dstPath, test, 0, 1) :
		return False
	if not RunTest(args, dstPath, test, 1, 1) :
		return False
	if test.batch == "1" :
		if not RunTest(args, dstPath, test, 1, 2) :
			return False
	return True

###################################################################################################

def main():

	parser = argparse.ArgumentParser(prog="Check", description="Synet tests check script.")
	parser.add_argument("-s", "--src", help="Tests data path.", required=False, type=str, default="./data")
	parser.add_argument("-b", "--bin", help="Tests binary path.", required=False, type=str, default="./build")
	parser.add_argument("-d", "--dst", help="Output tests path.", required=False, type=str, default="../test/check")
	parser.add_argument("-t", "--threads", help="Tests threads number.", required=False, type=int, default=1)
	parser.add_argument("-f", "--fast", help="Fast check flag (no conversion, small image count).", required=False, type=bool, default=False)
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, type=str, default="")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, type=str, default="")
	args = parser.parse_args()
	
	if not os.path.isdir(args.src):
		print("Test data directory '", args.src, "' is not exist!")
		return 1

	if not os.path.isdir(args.bin):
		print("Binary directory '", args.bin, "' is not exist!")
		return 1
	
	now = datetime.now();
	dstPath = "{0}_{1:04d}_{2:02d}_{3:02d}__{4:02d}_{5:02d}".format(args.dst, now.year, now.month, now.day, now.hour, now.minute)
	print("Test output directory: ", dstPath)
	if not os.path.isdir(dstPath):
		try:
			os.makedirs(dstPath)
		except OSError as error: 
			print("Can't create output directory '" , dstPath, "' !")
			return 1
	
	print("Synet Check:", args)	
	
	result : bool
	result, tests = GetTestList(args)
	if not result:
		return 1
	
	count = 0
	for test in tests:
		print("Test {}: {} {} {} {} {} ".format(count, test.framework, test.group, test.name, test.image, test.batch))
		if not RunTests(args, dstPath, test):
			return 1
		count += 1

	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
