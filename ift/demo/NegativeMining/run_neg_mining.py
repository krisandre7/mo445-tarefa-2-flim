import json
import os
import random
import sys
import time

import neg_mining.exp as exp
from neg_mining.error import Error



def main():
    if (len(sys.argv) == 1):
        Error.print_error("Usage: run_neg_mining.py <params01.json> <params02.json> ...", "main")
    
    t1 = time.time()
    for i in range(1, len(sys.argv)):
        json_file = open(sys.argv[i])

        try:
            params = json.load(json_file)
        except (ValueError, KeyError, TypeError):
            Error.print_error("JSON format error", "main")

        # seed = int(time.time())
        # print("\n*** seed = {0} ***\n".format(seed))
        # random.seed(seed)

        exp.run_exp(params)

        json_file.close()


    print("\n********* Total Time elapsed: {0:.2f} secs *********".format(time.time()-t1))
    print("\nDone...")




if __name__ == "__main__":
    sys.exit(main())