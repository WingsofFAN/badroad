import os
import json
import torch
import cv2 as cv
import numpy as np



def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    CLASSES = ['pavement defect']
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = [0, 255, 0]
    cv.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




class Solver:
    def __init__(self, args):
        self.args = args
        self.model= cv.dnn.readNetFromONNX(args.ckpt)

        self.output_file = self.args.format in ['txt', 'both']
        self.output_bbox = self.args.format in ['vis', 'both']


    
    def __call__(self):
        assert os.path.exists(self.args.source), f"{self.args.source} does not exist"

        if self.args.source.endswith('.list') or self.args.source.endswith('.txt'):
            with open(self.args.source) as f:
                lines = f.readlines()

            for line in lines:
                self.predict(line.strip())

        elif self.args.source.endswith('.png') or self.args.source.endswith('.jpg'):
            self.predict(self.args.source)

        else:
            print('unrecognizable source')
            print("source should be in one of the following format.")
            print("1. a file ended with '.list' or '.txt' containing lines of image paths")
            print("2. or a path to an image ended with '.jpg' or '.png")

            exit()



    def predict(self, path_to_im):
        im = cv.imread(path_to_im)
        self.im = im
        self.filename = path_to_im.split('/')[-1]

        blob = self.pre_process(im.copy())
        self.model.setInput(blob)
        outputs = self.model.forward()
        outputs = np.array([cv.transpose(outputs[0])])
        outputs = self.post_process(outputs)
        self.output(outputs)


    
    def output(self, dets):
        print(dets)

        if self.output_bbox:
            for det in dets:
                class_id, score, x0, y0, x1, y1 = det
                draw_bounding_box(self.im, class_id, score, x0, y0, x1, y1)
                
            cv.imwrite(os.path.join(self.args.output, f'{self.filename}'), self.im)
            
        if self.output_file:
            with open(os.path.join(self.args.output, f'{self.filename}.json'), 'w') as f:
                for det in dets:
                    f.write(json.dumps(det) + '\n')
            


    def pre_process(self, img):

        [height, width, _] = img.shape
        length = max((height, width))
        self.scale = length / 640

        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = img
        
        
        blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

        return blob




    def post_process(self, outputs):
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.3, 0.5)

        detections = []

        scale = self.scale 
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]

            detections.append([class_ids[index], 
                               scores[index], 
                               round(box[0] * self.scale ),
                               round(box[1] * self.scale ),
                               round((box[0] + box[2]) * self.scale ),
                               round((box[1] + box[3]) * self.scale )])

        return detections

                    


