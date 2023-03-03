def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script

import eel, os
from tkinter import *
from tkinter import filedialog

import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg

#import torchaudio.lib.libtorchaudio


eel.init('web')


@eel.expose
def output():
    # output_path = filedialog.askopenfilename(title="Output Image Location",initialdir='/', filetypes= (("jpg","*.jpg"),
    #                                               ("all files","*.*")))

    output_folder = filedialog.askdirectory(title="Output Image Location",initialdir='/')
    output_path = os.path.join(output_folder, "output.jpg")

    return output_path

@eel.expose
def input():
    input_path = filedialog.askopenfilename(title="Input Image Location",initialdir='/', filetypes= (("jpg","*.jpg"),
                                                  ("all files","*.*")))
    return input_path

@eel.expose
def Browse(value):
    path, output_path, model_path = value
    threshold = 0.7
    model_type = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file(model_type))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_path # Set path model .pth
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                metadata=None, 
                scale=1, 
                instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    im2 = Image.open(output_path)
    im2.show()
    return output_path

eel.start('file_access.html', size=(650, 400))

