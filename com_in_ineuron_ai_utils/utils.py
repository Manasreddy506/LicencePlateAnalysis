import base64
from logger import getLog

logger=getLog('utils.py')

def decodeImage(imgstring, fileName):

    try:

        imgdata = base64.b64decode(imgstring)
        logger.info("Image Decoded Succesfully")
        with open(fileName, 'wb') as f:
            f.write(imgdata)
            f.close()
        logger.info("Image Saved Sucessfully")

    except Exception as e:

        logger.exception(f"Failed to decode Image : \n{e}")
        raise Exception("Failed to decode Image")

def encodeImageIntoBase64(croppedImagePath):

    try:
        
        with open(croppedImagePath, "rb") as f:
            logger.info("Image encoded successfully")
            return base64.b64encode(f.read())
            
    except Exception as e:

        logger.exception(f"Failed to encode Image : \n{e}")
        raise Exception("Failed to encode Image")