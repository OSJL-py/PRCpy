import re
import os

def check_and_create_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)

        print("########################################################################")
        print("########################## SAVE PATH CREATED ##########################")
        print("########################################################################")

def check_file_exists(path):

    if os.path.exists(path):

        print("File already exists")

        return True

    else:

        return False

