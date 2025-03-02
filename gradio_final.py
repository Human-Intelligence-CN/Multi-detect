import torch
import gradio as gr


title = "基于gradio的目标检测"

description = "本项目基于gradio进行实现，点击左侧，选择待检测的图片，也可在下方选择实例图片进行预测"

base_conf, base_iou = 0.25, 0.45

model = torch.hub.load("./", "custom", path="runs/train/exp/weights/best.pt", source ="local")

def image_det(img,conf,iou):
    model.conf = conf
    model.iou = iou
    results = model(img)
    return results.render()[0]

gr.Interface(fn=image_det,
             inputs=["image",gr.slider(0,1,value=base_conf),gr.slider(0,1,base_iou)],
             outputs=["image"],
             live=True,
