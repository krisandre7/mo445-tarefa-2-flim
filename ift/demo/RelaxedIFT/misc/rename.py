import os
import sys

if len(sys.argv) != 3:
	print "<folder> <class>"
	sys.exit(1)


folder = sys.argv[1]
class_ = int(sys.argv[2])
files = [f for f in os.listdir(folder) if f.endswith(".scn")]
files.sort()
for file in files:
	number = file.split("_")[1]
	number = int(number[:number.find(".")])
	new_name = "%06d_%06d.scn" % (class_, number)
	os.rename(os.path.join(folder, file), os.path.join(folder, new_name))