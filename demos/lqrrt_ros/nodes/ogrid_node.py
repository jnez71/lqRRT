#!/usr/bin/env python
"""
Opens a neat window for drawing occupancy grids!

"""
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData

rospy.init_node("ogrid_node")

class DrawGrid(object):
    def __init__(self, height, width):
        self.height, self.width = height, width

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.do_draw)
        self.clear_screen()
        self.drawing = 0

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
    def __init__(self):
        height = int(rospy.get_param("~grid_height", 800))
        width = int(rospy.get_param("~grid_width", 800))
        resolution = rospy.get_param("~grid_resolution", .25)
        ogrid_topic = rospy.get_param("/lqrrt_node/ogrid_topic", "/ogrid")

        self.grid_drawer = DrawGrid(height, width)
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

o = OGridPub()

while True:
    cv2.imshow("image", o.grid_drawer.img)
    k = cv2.waitKey(100) & 0xFF

    if k == 27:
        break
    elif k == 113:
        o.grid_drawer.clear_screen()

cv2.destroyAllWindows()
