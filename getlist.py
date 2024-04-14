from pathlib import Path

def main():
    rootPath = Path(r"/home/fan/yolov8/road/images")
    imglist = rootPath.rglob("*jpg")

    with open("train.txt",'w') as f:
        for p in imglist:
            f.write(str(p)+"\n")

if __name__ == "__main__":
    main()