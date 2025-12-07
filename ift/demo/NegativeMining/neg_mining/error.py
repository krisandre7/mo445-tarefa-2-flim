"""
Error messages.
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 9, 2015

import sys

class Error(object):
    @staticmethod
    def print_error(error_msg, function):
        sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))