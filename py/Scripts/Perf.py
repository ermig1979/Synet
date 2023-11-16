import argparse
import os
import datetime
import threading
import multiprocessing
import subprocess
import random

###################################################################################################

class Test():
	def __init__(self, vals):
		self.framework = vals[0]
		self.group = vals[1]
		self.name = vals[2]
		self.image = vals[3]
		self.batch = vals[4]
		self.path = ""
		
###################################################################################################

class Context():
	def __init__(self, args):
		self.args = args
		self.tests = []
		self.total = 0
		self.curr = 0
		self.dst = ""

	def UpdateProgress(self, log) :
		self.curr += 1;
		print("Test {0} of {1} log to : {2}".format(self.curr, self.total, log))		
			
	def Error(self, text="") -> bool :
		if text != "" :
			print(text)
		return False
	
###################################################################################################

def ValidateParameters(context : Context):
	args = context.args
	if args.framework == "i" :
		args.framework = "inference_engine"
	elif args.framework == "o" :
		args.framework = "onnx"
	elif args.framework == "q" :
		args.framework = "quantization"
	else :
		print("Unknown parameter -f={0}!".format(args.framework))
		return False
	
	if args.threads == 0 :
		args.threads = 1
	if args.threads < 0 :
		args.threads = max(1, multiprocessing.cpu_count() // abs(args.threads))
	return True

###################################################################################################

def CheckDirs(context : Context) :
	args = context.args
	
	if not os.path.isdir(args.src):
		return context.Error("Test data directory '{0}' is not exist!".format(args.src))
	
	if not os.path.isdir(args.bin):
		return context.Error("Binary directory '{0}' is not exist!".format(args.bin))
	
	context.dst = args.dst + os.path.sep
	now = datetime.datetime.now()
	context.dst += "{0:04d}_{1:02d}_{2:02d}__{3:02d}_{4:02d}{5}_t{6}".format(now.year, now.month, now.day, now.hour, now.minute, args.framework[0], args.threads)
	if len(args.include) > 0 :
		context.dst += "_dbg"
	if not os.path.isdir(context.dst):
		try:
			os.makedirs(context.dst)
		except OSError as error: 
			return context.Error("Can't create output directory '{0}' !".format(context.dst))
	
	return True
		
###################################################################################################

def SetTestList(context : Context):
	args = context.args
	listPath = args.src + os.path.sep + "tests.txt"
	if not os.path.isfile(listPath):
		return context.Error("Test list '{}' is not exist!".format(listPath))
	listFile = open(listPath, "r")
	while True:
		line = listFile.readline()
		if not line:
			break
		vals = line.split()
		if len(vals) < 5 :
			continue
		if vals[0][0] == '#' :
			continue
		test = Test(vals)
		
		if test.framework != args.framework :
			continue
		
		test.path = test.framework
		if test.group != "root" :
			test.path += os.path.sep + test.group
		test.path += os.path.sep + test.name
		
		if len(args.include) > 0 :
			skip = True
			for include in args.include :
				if test.path.find(include) != -1 :
					skip = False
			if skip :
				continue
			
		if len(args.exclude) > 0 :
			skip = False
			for exclude in args.exclude :
				if test.path.find(exclude) != -1 :
					skip = True
			if skip :
				continue
		
		if test.batch == "1" :
			context.total += 2
		else :
			context.total += 1
		context.tests.append(test)
	listFile.close()
	if len(context.tests) == 0 :
		return context.Error("There are no tests found for given include({0}) and exclude({1}) filters in file '{2}'!".format(args.include, args.exclude, listPath))
	return True

###################################################################################################

def RunTest(context, test, batch):
	args = context.args
	binPath = args.bin + os.path.sep + "test_" + test.framework
	if not os.path.isfile(binPath):
		return context.Error("Binary '{0}' is not exist!".format(binPath))
	
	testPath = args.src + os.path.sep + test.path
	if not os.path.isdir(testPath):
		return context.Error("Test directory '{0}' is not exist!".format(testPath))
	
	imagePath = ""
	if test.image == "local" :
		imagePath = testPath + os.path.sep + "image"
	else :
		imagePath = args.src + os.path.sep + "images" + os.path.sep + test.image
	if not os.path.isdir(imagePath):
		return context.Error("Image directory '{0}' is not exist!".format(imagePath))
	
	log = ""
	if test.group == "root" :
		log = context.dst + os.path.sep + "p{0}_{1}_t{2}_b{3}.txt".format(test.framework[0], test.name, args.threads, batch)
	else :
		log = context.dst + os.path.sep + "p{0}_{1}__{2}_t{3}_b{4}.txt".format(test.framework[0], test.group, test.name, args.threads, batch)

	pathArgs = ""
	if args.framework == "quantization" :
		threshold = 0.02
		pathArgs += "-fm={0}/synet.xml -fw={0}/synet.bin -sm={0}/int8.xml".format(testPath)
	else:
		threshold = 0.0031
		if test.framework == "inference_engine" :
			pathArgs += "-fm={0}/other.xml -fw={0}/other.bin".format(testPath)
		elif test.framework == "onnx" :
			pathArgs += "-fw={0}/other.onnx".format(testPath)
		pathArgs += " -sm={0}/synet.xml".format(testPath)
	pathArgs += " -sw={0}/synet.bin -id={1} -od={0}/output -tp={0}/param.xml -sn={2}/sync.txt -hr={2}/_report.html -tr={2}/_report.txt".format(testPath, imagePath, context.dst)
	
	trashFile = imagePath + os.path.sep + "descript.ion"
	if os.path.isfile(trashFile) :
		os.remove(trashFile)
	
	context.UpdateProgress(log)
	
	if batch == 1 :
		cmd = "{0} -m=convert {1} -tf=1 -cs=1".format(binPath, pathArgs)
		result = subprocess.run(cmd.split())
		if result.returncode != 0 :
			return context.Error("Error in test {0} !".format(log))
		
	cmd = "{0} -m=compare -e=3 {1} -rn=0 -et=10.0 -wt=1 -tt={2} -ie=10 -be=10 -tf=1 -bs={3} -ct={4} -cs=1 -ln={5}".format(binPath, pathArgs, args.threads, batch, threshold, log)
	result = subprocess.run(cmd.split())
	if result.returncode != 0 :
		return context.Error("Error in test {0} !".format(log))

	return True

###################################################################################################

def RunAllTests(context):
	for test in context.tests:
		if not RunTest(context, test, 1) :
			return False
		if test.batch == "1" :
			if not RunTest(context, test, 10) :
				return False
	return True

###################################################################################################

def main():

	parser = argparse.ArgumentParser(prog="Perf", description="Synet performance test script.")
	parser.add_argument("-s", "--src", help="Tests data path.", required=False, type=str, default="./data")
	parser.add_argument("-b", "--bin", help="Tests binary path.", required=False, type=str, default="./build")
	parser.add_argument("-d", "--dst", help="Output tests path.", required=False, type=str, default="../test/perf")
	parser.add_argument("-t", "--threads", help="Tests threads number.", required=False, type=int, default=1)
	parser.add_argument("-f", "--framework", help="Framework to test. It can be i(inference_engine), o(onnx), or q(quantization).", required=False, type=str, default="o", choices=["i", "o", "q"])
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, default=[], action="append")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, default=[], action="append")
	context = Context(parser.parse_args())
	
	ValidateParameters(context)
	
	if not CheckDirs(context) :
		return 1
	
	if not SetTestList(context):
		return 1
	
	print("\nSynet start performance tests in {0} threads: \n".format(context.args.threads))
	
	start = datetime.datetime.now()
	
	if not RunAllTests(context) :
		return 1
		
	elapsed = datetime.datetime.now() - start
	
	print("All tests finished successfully in {0}:{1:02d}:{2:02d} !\n".format(elapsed.seconds // 3600, elapsed.seconds % 3600 // 60, elapsed.seconds % 60))
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
