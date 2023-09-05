import argparse
import os

###################################################################################################

def GetTestList(args):
	listPath = args.src + "/tests.txt"
	print(listPath)
	if not os.path.isfile(listPath):
		print("Test list '", listPath, "' is not exist!")
		return False
	listFile = open(listPath, "r")
	count = 0
	while True:
		line = listFile.readline()
		if not line:
			break
		vals = line.split()
		if len(vals) < 5 :
			continue
		print("Test {}: {}".format(count, line.strip()))
		count += 1
	listFile.close()
	return True

###################################################################################################

def main():

	parser = argparse.ArgumentParser(prog="Check", description="Synet tests check script.")
	parser.add_argument("-s", "--src", help="Tests data path.", required=False, type=str, default="./data")
	parser.add_argument("-d", "--dst", help="Output tests path.", required=False, type=str, default="../test")
	parser.add_argument("-t", "--threads", help="Tests threads number.", required=False, type=int, default=1)
	parser.add_argument("-f", "--fast", help="Fast check flag (no conversion, small image count).", required=False, type=bool, default=False)
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, type=str, default="")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, type=str, default="")
	args = parser.parse_args()
	
	print("Synet Check:", args)	
	
	GetTestList(args)
	
###################################################################################################
	
if __name__ == "__main__":
	main()
