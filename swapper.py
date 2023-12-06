"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""

import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_many_faces(face_analyser,
                   frame:np.ndarray,
                   num_faces):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame, num_faces)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper,
              source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process(source_img: Union[Image.Image, List],
            target_img: Image.Image):

    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in the target image
    num_faces = len(source_img)
    target_faces = get_many_faces(face_analyser, target_img, num_faces)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        print("Replacing faces in target image from the left to the right by order")
        for i in range(num_faces):
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR), 1)
            source_index = i
            target_index = i

            if source_faces is None:
                raise Exception("No source faces found!")

            temp_frame = swap_face(
                face_swapper,
                source_faces,
                target_faces,
                source_index,
                target_index,
                temp_frame
            )
        result = temp_frame
    else:
        print("No target faces found!")

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

# load machine default available providers
providers = onnxruntime.get_available_providers()

model = "./checkpoints/inswapper_128.onnx"
# load face_analyser
face_analyser = getFaceAnalyser(model, providers)

# load face_swapper
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
face_swapper = getFaceSwapModel(model_path)