import os

def getAbsPath(relativePath):
    output = os.path.abspath( os.path.join( os.path.dirname( __file__ ), relativePath) )
    if (os.path.exists(output)):
        return output
    else:
        raise Exception(f"Path <{relativePath}> relative to Constants.py does not exis")


FOLDER_DATA_DESCRIPTORS = getAbsPath("../data/")