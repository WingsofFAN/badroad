import torch
import argparse
from PIL import Image
from model import Solver


import cv2 as cv
import glob

def main(args):
    solver = Solver(args)
    solver()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help='path to pytorch model.', default="yolov8.onnx")
    parser.add_argument("-f", "--format", help='output format.]' , default='both')
    parser.add_argument("-s", "--source", help='files for inference.',default=r"/home/SENSETIME/yangfan5/code/yolo/badroad/road/images/train/2020_07_17_09_17_34.jpg")
    parser.add_argument("-o", "--output", help='dir to save output result]',default="/home/SENSETIME/yangfan5/code/yolo/badroad/TRI_road_infer/output")
    args = parser.parse_args()

    main(args)
    
    
