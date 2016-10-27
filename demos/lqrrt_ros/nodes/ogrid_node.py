#!/usr/bin/env python
"""
Opens a neat window for drawing occupancy grids!

"""
import rospy
import cv2
import numpy as np
import argparse
import sys

from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData

rospy.init_node("ogrid_node")

class DrawGrid(object):
    def __init__(self, height, width, image_path):
        self.height, self.width = height, width

        self.clear_screen()

        if image_path:
            self.make_image(image_path)

        cv2.namedWindow('Draw OccupancyGrid')
        cv2.setMouseCallback('Draw OccupancyGrid', self.do_draw)
        self.drawing = 0

    def make_image(self, image_path):
        img = cv2.imread(image_path, 0)
        if img is None:
            print "Image not found at '{}'".format(image_path)
            return

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        self.img = np.clip(img, -1, 100)

    def clear_screen(self):
        self.img = np.zeros((self.height, self.width), np.uint8)

    def do_draw(self, event, x, y, flags, param):
        draw_vals = {1: 100, 2: 0}

        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = 0
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = 2
        elif self.drawing != 0:
            cv2.circle(self.img, (x, y), 5, draw_vals[self.drawing], -1)

class OGridPub(object):
    def __init__(self, image_path=None):
        height = int(rospy.get_param("~grid_height", 800))
        width = int(rospy.get_param("~grid_width", 800))
        resolution = rospy.get_param("~grid_resolution", .25)
        ogrid_topic = rospy.get_param("/lqrrt_node/ogrid_topic", "/ogrid")

        self.grid_drawer = DrawGrid(height, width, image_path)
        self.ogrid_pub = rospy.Publisher(ogrid_topic, OccupancyGrid, queue_size=1)

        m = MapMetaData()
        m.resolution = resolution
        m.width = width
        m.height = height
        pos = np.array([-width * resolution / 2, -height * resolution / 2, 0])
        quat = np.array([0, 0, 0, 1])
        m.origin = Pose()
        m.origin.position.x, m.origin.position.y = pos[:2]
        self.map_meta_data = m

        rospy.Timer(rospy.Duration(1), self.pub_grid)

    def pub_grid(self, *args):
        grid = self.grid_drawer.img

        ogrid = OccupancyGrid()
        ogrid.header.frame_id = '/world'
        ogrid.info = self.map_meta_data
        ogrid.data = np.subtract(np.flipud(grid).flatten(), 1).astype(np.int8).tolist()

        self.ogrid_pub.publish(ogrid)


usage_msg = "Lets you draw an occupancy grid!"
desc_msg = "If you pass -i <path to image>, the ogrid will be created based on thresholded black and white image."

parser = argparse.ArgumentParser(usage=usage_msg, description=desc_msg)
parser.add_argument('-i', '--image', action='store', type=str,
                    help="Path to an image to use as an ogrid.")

# Roslaunch passes in some args we don't want to deal with (there may be a better way than this
if '-i' in sys.argv:
    args = parser.parse_args(sys.argv[1:])
    im_path = args.image
else:
    im_path = None

o = OGridPub(image_path=im_path)

while True:
    cv2.imshow("Draw OccupancyGrid", o.grid_drawer.img)
    k = cv2.waitKey(100) & 0xFF

    if k == 27:
        break
    elif k == 113:
        o.grid_drawer.clear_screen()

cv2.destroyAllWindows()
