import sys
sys.path.append('./CodeFormer/CodeFormer')

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PIL import Image

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

def check_ckpts():
  pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
  }
  # download weights
  if not os.path.exists('CodeFormer/CodeFormer/weights/CodeFormer.pth'):
    load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='CodeFormer/CodeFormer/weights/CodeFormer', progress=True, file_name=None)
  if not os.path.exists('CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
    load_file_from_url(url=pretrain_model_url['detection'], model_dir='CodeFormer/CodeFormer/weights/facelib', progress=True, file_name=None)
  if not os.path.exists('CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth'):
    load_file_from_url(url=pretrain_model_url['parsing'], model_dir='CodeFormer/CodeFormer/weights/facelib', progress=True, file_name=None)


def face_restoration(img, codeformer_fidelity):
  """Run a single prediction on the model"""
  try: # global try
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    face_helper.all_landmarks_5 = []
    face_helper.det_faces = []
    face_helper.affine_matrices = []
    face_helper.inverse_affine_matrices = []
    face_helper.cropped_faces = []
    face_helper.restored_faces = []
    face_helper.pad_input_imgs = []
    face_helper.read_image(img)
    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(
      blur_ratio=0
    )
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
      # prepare data
      cropped_face_t = img2tensor(
        cropped_face / 255.0, bgr2rgb=True, float32=True
      )
      normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
      cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

      try:
        with torch.no_grad():
          output = codeformer_net(
            cropped_face_t, w=codeformer_fidelity, adain=True
          )[0]
          restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        del output
        torch.cuda.empty_cache()
      except RuntimeError as error:
        print(f"Failed inference for CodeFormer: {error}")
        restored_face = tensor2img(
          cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
        )

      restored_face = restored_face.astype("uint8")
      face_helper.add_restored_face(restored_face)

    # paste_back
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(
      upsample_img=None, draw_box=False
    )

    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_img)
  except Exception as error:
    print('Global exception', error)
    return None, None

# make sure the ckpts downloaded successfully
check_ckpts()
# https://huggingface.co/spaces/sczhou/CodeFormer
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                  codebook_size=1024,
                                                  n_head=8,
                                                  n_layers=9,
                                                  connect_list=["32", "64", "128", "256"],
                                                ).to(device)
ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

# take the default setting for the demo
detection_model = "retinaface_resnet50"
upscale = 1

face_helper = FaceRestoreHelper(
    upscale,
    crop_ratio=(1, 1),
    det_model=detection_model,
    save_ext="png",
    use_parse=False,
)