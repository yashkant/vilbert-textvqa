# import numpy as np
# from math import sqrt
# from shapely.geometry import Polygon
#
#
# def distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#
# def iou(rect1, rect2):
#     """Calculates IoU between two shapely polygons"""
#     x1, y1, x2, y2 = rect1
#     a1, b1, a2, b2 = rect2
#     poly1 = Polygon(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float))
#     poly2 = Polygon(np.array([[a1, b1], [a2, b1], [a2, b2], [a1, b2]], dtype=float))
#     return poly1.intersection(poly2).area / poly1.union(poly2).area
#
#
# def centroid(rect):
#     x1, y1, x2, y2 = rect
#     return (x1 + x2) / 2, (y1 + y2) / 2
#
#
# def contains(rect1, rect2):
#     x1, y1, x2, y2 = rect1
#     a1, b1, a2, b2 = rect2
#     if ((x1 - a1) <= 0) and ((y1 - b1) <= 0) and ((x2 - a2) >= 0) and ((y2 - b2) >= 0):
#         return True
#     else:
#         return False
#
#
# def angle(rect1, rect2):
#     """
#     We draw x-axis with reference to point1.
#     Returns value between [0, 360] degrees.
#     """
#     point1 = centroid(rect1)
#     point2 = centroid(rect2)
#     del_y = point2[1] - point1[1]
#     del_x = point2[0] - point1[0]
#     # signed angle (https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html)
#     ang = np.arctan2(del_y, del_x)
#     return np.rad2deg(ang % (2 * np.pi))
#
#
# def classify(rect1, rect2):
#     """
#     Returns classes for two edges from rect1 -> rect2 and rect2 -> rect1.
#     rect1: xyxy
#
#     Classes:
#     Covers - 0
#     Inside - 1
#     Overlap - 2
#     Quad1-1 - 3
#     Quad1-2 - 4
#     Quad2-1 - 5
#     Quad2-2 - 6
#     Quad3-1 - 7
#     Quad3-2 - 8
#     Quad4-1 - 9
#     Quad4-2 - 10
#     None - 11
#     """
#
#     # check for covers/inside:
#     if contains(rect1, rect2):
#         return 0,1
#
#     # check for covers/inside:
#     if contains(rect2, rect1):
#         return 1,0
#
#     _iou = iou(rect1, rect2)
#
#     if _iou < 0.5:
#
#
#
#
#     import pdb
#     pdb.set_trace()
#
#
#
#
# # def tangle(point1, point2):
# #     """
# #     We draw x-axis with reference to point1.
# #     """
# #     del_y = point2[1] - point1[1]
# #     del_x = point2[0] - point1[0]
# #     # signed angle (https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html)
# #     ang = np.arctan2(del_y, del_x)
# #     return np.rad2deg(ang % (2 * np.pi))
#
#
# if __name__ == "__main__":
#     rect1 = np.array([0, 0, 1, 1], dtype=float)
#     rect2 = np.array([0.5, 0.5, 1.5, 1.5])
#     rect3 = np.array([0.25, 0.25, 0.75, 0.5])
#     rect4 = np.array([1, 0.25, 1.75, 1.25])
#     p1 = np.array([0, 0], dtype=float)
#     p2 = np.array([1, 1], dtype=float)
#     p3 = np.array([-1, -1], dtype=float)
#     p4 = np.array([1, -1], dtype=float)
#     p5 = np.array([-1, 1], dtype=float)
#     import pdb
#     pdb.set_trace()
#     iou(rect1, rect2)
