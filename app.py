from swapper import *
from restoration import *
import gradio as gr
#import time

def swap(source_img, target_img):
  #start_time = time.time()
  result_image = process([source_img], target_img)
  #print("--- swap %s seconds ---" % (time.time() - start_time))
  #result_image.save("./data/result0.png")

  #start_time = time.time()
  result_image = face_restoration(result_image, 1)
  #print("--- restore %s seconds ---" % (time.time() - start_time))

  return result_image

demo = gr.Interface(
  fn=swap,
  inputs=[gr.Image(), gr.Image()],
  outputs=["image"],
)
demo.queue().launch()
#swap(Image.open("./data/t.jpg"), Image.open("./data/s.jpg")).save("./data/result.png")
