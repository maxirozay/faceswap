from swapper import *
from restoration import *
import gradio as gr

def swap(source_img, target_img):
  result_image = process([source_img], target_img)
  #result_image.save("./data/result0.png")

  result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
  result_image = face_restoration(result_image, 1)
  result_image = Image.fromarray(result_image)

  return result_image

demo = gr.Interface(
  fn=swap,
  inputs=[gr.Image(), gr.Image()],
  outputs=["image"],
)
demo.launch(root_path="/faceswap")
#swap(Image.open("./data/s.jpg"), Image.open("./data/t.jpg")).save("./data/result.png")
