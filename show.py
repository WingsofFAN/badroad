import cv2
import json
from pathlib import Path

def getrect( path ):
    with open(str(path).replace("jpg","json"),'r') as f:
        data = json.load(f)
        rects = []
        for p in data["shapes"]:
            rects.append(p["points"])

    return rects

def main():
    rootpath = Path("labelme")
    imglist = rootpath.rglob("*jpg")

    res = {}
    with open("runs/detect/val2/predictions.json",'r') as f:
        data = json.load(f)
        for p in data:
            if p["image_id"] not in res.keys():
                res[p["image_id"]] = []
            res[p["image_id"]].append(p["bbox"])
    t = 0
    for imgp in imglist:
        img = cv2.imread(str(imgp))
        rects = getrect(imgp)

        for p in rects:
            cv2.rectangle(img,(int(p[0][0]),int(p[0][1])),(int(p[1][0]),int(p[1][1])),(0,0,255),3)
        id = str(imgp)[8:-4]
        if id not in res.keys():
            continue
        for p in res[id]:
            cv2.rectangle(img,(int(p[0]),int(p[1])),(int(p[0]+p[2]),int(p[1]+p[3])),(0,255,0),1)
        
        cv2.imwrite("/home/fan/yolov8/res/"+str(t)+".jpg",img)
        t  = t + 1
        # cv2.imshow("img",img)
        # key = cv2.waitKey(0)
        # if key == 32:
        #     cv2.imwrite(str(t)+".jpg",img)
        #     


if __name__ == "__main__":
    main()

