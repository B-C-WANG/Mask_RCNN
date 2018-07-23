import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config

import cv2
import time


class RealTimeMaskRCNN():
    def __init__(self):
        # Root directory of the project
        ROOT_DIR = "C:\\Users\Administrator\Desktop\Mask_RCNN"

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        import coco
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory of images to run detection on
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']
        self.ax = []
        self.plt = []
        self.cap = cv2.VideoCapture(0)

    def detect_one_image_camera(self):
            cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            results = self.model.detect([frame], verbose=1)
            r = results[0]
            ax, plt = visualize.display_instances(frame,
                                                  r['rois'], r['masks'], r['class_ids'],
                                                  self.class_names, r['scores'],
                                                  )
            show_time = 2
            if show_time:
                plt.ion()
                plt.show()
                plt.pause(show_time)
                plt.close()
            else:
                plt.show()
            print("image showed")
        # # Visualize results
        # r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])
    def get_camera_image(self):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise  RuntimeError("Unable to read camera feed")
            #
            ret, frame = cap.read()
            print(ret,frame)
            #frame = cv2.imread("12283150_12d37e6389_z.jpg")
            results = self.model.detect([frame], verbose=1)
            r = results[0]
            print(self.class_names)

            ax, plt = visualize.display_instances(image=frame,
                                                  boxes=r['rois'],
                                                  masks=r['masks'],
                                                  class_ids=r['class_ids'],
                                                  class_names=self.class_names,
                                                  scores=r['scores']
                                                  )
            self.ax.append(ax)
            self.plt.append(plt)

            return ax,plt
    def thread_get_camera_image(self):
        #while 1: # or use: when show, will delete the ax and plt, and when ax and plt pool is more than 5, wait, less than 5, prepare!
        while 1:

                if len(self.plt) >= 10:
                    print("### wait for showing")
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                results = self.model.detect([frame], verbose=1)
                r = results[0]
                ax, plt = visualize.display_instances(frame,
                                                      r['rois'], r['masks'], r['class_ids'],
                                                      self.class_names, r['scores'],
                                                      )
                self.ax.append(ax)
                self.plt.append(plt)
                print(self.ax)

    def thread_run_time_show(self):
        for i in range(100):
            print("showing__________________")
            #ax, plt = self.get_camera_image()
            try:
                ax = self.ax.pop(-1)
                plt = self.plt.pop(-1)
                print("____________show image get ax: ___________",ax)
            except:
                time.sleep(1)# wait for another thread to prepare image
                print("no image")
                continue
            ax.set_aspect('auto')
            #plt.subplots_adjust(wspace=0, hspace=0)
            #plt.ion()
            plt.show()
            #plt.pause(0.5)
            plt.close()
            print("image showed")

    def test_thread(self):
        # TODO: make prepare image and show image different thread
        #prepare = Thread(target=self.thread_get_camera_image)
        #show = Thread(target=self.thread_run_time_show)

        #prepare.start()
        #time.sleep(3)
        #show.start()
        self.thread_get_camera_image()
        print("running time show")
        self.thread_run_time_show()


    def run_time_show(self):
        for _ in range(100):
            ax,plt = self.get_camera_image()

            ax.set_aspect('auto')
            plt.subplots_adjust(wspace=0, hspace=0)
            #plt.ion()
            plt.show()
            #plt.pause(0.5)
            #plt.close()
            print("***************image showed")



    def detect_one_image(self,image_path):

        image = skimage.io.imread(image_path)
        results = self.model.detect([image],verbose=1)
        r = results[0]
        ax,plt = visualize.display_instances(image,
                                   r['rois'], r['masks'], r['class_ids'],
                                   self.class_names, r['scores'],
                                   )
        show_time=2
        if show_time:
            plt.ion()
            plt.show()
            plt.pause(show_time)
            plt.close()
        else:
            plt.show()
        print("image showed")
def test():
    temp = RealTimeMaskRCNN()
    #temp.detect_one_image_camera()
    temp.run_time_show()
    #temp.test_thread()
if __name__ == '__main__':
    test()