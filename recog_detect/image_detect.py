import matplotlib.pyplot as plt
import sys
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

args = sys.argv

if len(args)!=2:
    print('usage: python3 image_detect.py [img_path]')
    sys.exit(1)

img = read_image(args[1])
model = SSD300(pretrained_model='voc0712')
bboxes, labels, scores = model.predict([img])
vis_bbox(img, bboxes[0], labels[0], scores[0],
        label_names=voc_bbox_label_names)
plt.show()
