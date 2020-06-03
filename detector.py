from random import randint
import numpy as np
import cv2
from PIL import ImageColor

class Detector:

    dnn = None
    colors = []
    classes = []
    layer_names = None
    output_layers = None

    # Constructor
    def __init__(self):
        # Initialise DNN from file
        self.dnn = cv2.dnn.readNet(r"dnn\yolov3.weights", r"dnn\yolov3.cfg")
        
        # Initialise detected object names
        with open(r"dnn\coco.names", "r") as names:
            self.classes = [line.strip() for line in names.readlines()]

        # Initialise colors
        for i in range(len(self.classes)):
            self.colors.append('#%06X' % randint(0, 0xFFFFFF))
        del i
        
        # Initialise layer names
        self.layer_names = self.dnn.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.dnn.getUnconnectedOutLayers()]

    def detect(self, img):
        ''' Detects objects from image

        Method input is only image (img). For videos call this method every captured frame.
        Method uses YOLOv3 approache to object recognition. 
        DNN and pretrained weights set are property of pjreddie user form Github.

        '''
        # Get image 
        height, width, channels = img.shape
        del channels

        # Detect objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Push image to DNN
        self.dnn.setInput(blob)
        outs = self.dnn.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for entry in outs:
            for detection in entry:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detected with 40% chance
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Add to arrays
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])

            # Example of getting info about detected object. To be deleted
            if "person" in label: print("Human detected!")

            color = self.colors[class_ids[i]]
            # Check when to use white / black font
            red, green, blue = bytes.fromhex(color.lstrip('#'))
            hsp = np.sqrt(0.299 * (red * red) + 0.587 * (green * green) + 0.114 * (blue * blue))
            if hsp > 127.5:
                # Light
                font_color = (0, 0, 0)
            else:
                # Dark
                font_color = (255, 255, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), ImageColor.getrgb(color), int(confidences[i]*4))
            # Draw rectangle under text:
            # Get text size and make box out of it
            temp = str(confidences[i]*100)
            text = label + " " + temp[0:5] + "%"
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, thickness=1)[0]
            box_coords = ((x, y), (x + text_width + 2, y - text_height - 2))
            cv2.rectangle(img, box_coords[0], box_coords[1], ImageColor.getrgb(color), cv2.FILLED)

            cv2.rectangle(img, (x, y), (x+60, y-10), ImageColor.getrgb(color), -1)
            cv2.putText(img, text, (x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=font_color, thickness=1)
    
        return img
