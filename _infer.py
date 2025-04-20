from typing import Any, Optional
from functools import lru_cache
import cv2
import numpy
import torch
import onnxruntime
import os

Frame = numpy.ndarray[Any, Any]
detector = onnxruntime.InferenceSession("_model.onnx", providers=["CUDAExecutionProvider"])

def readFile(filePath):
    content = ""

    with open(filePath, "r") as file:
        content = file.read()

    return content

def prepare_frame(frame : Frame) -> Frame:
	frame = cv2.resize(frame, (224, 224)).astype(numpy.float32)
	frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	frame = numpy.expand_dims(frame, axis = 0)
	return frame

def analyse_frame(frame : Frame) -> bool:
	frame = prepare_frame(frame)
	probability = detector.run(None,
	{
		'input:0': frame
	})[0][0][1]
	return probability > 0.80

def read_image(image_path : str) -> Optional[Frame]:
	if image_path:
		return cv2.imread(image_path)
	return None

def analyse_image(image_path : str) -> bool:
	frame = read_image(image_path)
	return analyse_frame(frame)

if analyse_image(readFile("input_file.txt")):
	with open("result.txt", "w") as file:
		file.write("1")
else:
	with open("result.txt", "w") as file:
		file.write("0")

with open("finished.txt", "w") as file:
	pass