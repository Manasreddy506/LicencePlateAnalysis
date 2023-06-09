import cv2
import os
import torch 
import numpy as np
from PIL import Image
from logger import getLog
from io import BytesIO
import matplotlib.pyplot as plt
from com_in_ineuron_ai_utils.utils import encodeImageIntoBase64
from paddleocr import PaddleOCR,draw_ocr
import cv2 as cv


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logger=getLog('Detector.py')

class Detector():

    def __init__(self):

        try:

            self.model = torch.hub.load('ultralytics/yolov5',"custom", path='my_model/best.pt')
            self.ocr=PaddleOCR(use_angle_cls=True)
            logger.info("Detector object initialized")
        
        except Exception as e:
            
            logger.exception(f"Failed to intialize Detector object : \n{e}")
            raise Exception("Failed to intialize Detector object")


    def run_inference(self):

        try:

            image_path = "inputImage.jpg"
            image = cv2.imread(image_path)
            sharpened_image = unsharp_mask(image)
            image = cv2.resize(sharpened_image, (540, 540)) 
            model = self.model
            predictions = model(image)
            data=predictions.pandas().xyxy[0]
            for index, row in data.iterrows():
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                cropped_image = image[y1:y2, x1:x2]
                result = self.ocr.ocr(cropped_image)
                if len(result)!=0:
                    txts = [line[1][0] for line in result]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, txts[0], (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2, cv2.LINE_AA)
            output_filename = 'output.jpg'
            cv2.imwrite(output_filename, image)
            logger.info("Output Image Saved")
            opencodedbase64 = encodeImageIntoBase64("output.jpg")
            result = {"image": opencodedbase64.decode('utf-8')}
            logger.info("Inference Completed")
            return result

        except Exception as e:

            logger.exception(f"Failed to complete inference : \n{e}")
            raise Exception("Failed to complete inference ")

