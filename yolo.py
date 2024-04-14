from ultralytics import YOLO
import cv2

def main():
#   # 训练加载模型
    # model = YOLO('/home/fan/yolov8/yolov8.yaml').load('/home/fan/yolov8/yolov8n.pt')  # build from YAML and transfer weights
    # model = YOLO('/home/fan/yolov8/yolov8.yaml').load('/home/fan/yolov8/yolov8n.pt')  # build from YAML and transfer weights
    # model = YOLO('/home/fan/yolov8/yolov8.yaml').load('/home/fan/yolov8/yolov8n.pt')  # build from YAML and transfer weights
    model = YOLO('/home/fan/yolov8/yolov8.yaml').load('/home/fan/yolov8/runs/detect/train5/weights/best.pt')  # build from YAML and transfer weights
    
    
    
    # 测试加载模型
    # model = YOLO('/home/fan/yolov8/yolov8.yaml').load("/home/fan/yolov8/runs/detect/train/weights/best.pt")

    # results = model.train(data='/home/fan/yolov8/road.yaml', epochs=200, imgsz=640)
    # validation_results = model.val(data='/home/fan/yolov8/road.yaml',
    #                            imgsz=640,
    #                            batch=16,
    #                            conf=0.3,
    #                            iou=0.5,
    #                            save_json=True,
    #                            save_hybrid=True,
    #                            device='0')
    pre = model.predict(source="/home/fan/yolov8/meta/all_val.txt",
                                conf=0.3,iou=0.5,
                                save=True,save_conf=True,
                                save_txt=True,name='output')

    return
    



if __name__ == "__main__":
    main()