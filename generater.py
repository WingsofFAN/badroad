from pathlib import Path 
import os

def main():
    metaPath = Path("/home/fan/yolov8/meta")
    listsP =  metaPath.rglob("*list")
    subfix = r"/home/fan/yolov8/road/images/"
    for p in listsP:
        saveName = str(p).replace("list","txt")
        new = []
        with open(str(p),'r') as f:
            data = f.readlines()
            for line in data:
                new.append(subfix+line)
        with open(saveName,'w') as f:
            for line in new:
                f.write(line)



def check():
    import glob
    root = '/home/fan/yolov8/meta'

    files = glob.glob(os.path.join(root, '*train*txt'))

    test, train = [], []
    for ptrain in files:
        ptest = ptrain.replace('train', 'test')

        with open(ptrain,'r') as f:
            train += f.readlines()
        
        with open(ptest, 'r') as f:
            test += f.readlines()

    train = set(train)
    for i in test:
        print(i in train)     
        
                        
                

if __name__ == "__main__":
#     main()
    check()