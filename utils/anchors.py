# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import math
from utils.bbox_helper import center2corner, corner2center


class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.round_dight = 0
        self.image_center = 0
        self.size = 0
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density**2)
        self.anchors = None  # in single position (anchor_num*4)
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density)*anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                if self.round_dight > 0:
                    ws = round(math.sqrt(size*1. / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size*1. / r))
                    hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w*0.5+x_offset, -h*0.5+y_offset, w*0.5+x_offset, h*0.5+y_offset][:]
                    count += 1

    def generate_all_anchors(self, im_c, size):
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


# if __name__ == '__main__':
#     anchors = Anchors(cfg={'stride':16, 'anchor_density': 2})
#     anchors.generate_all_anchors(im_c=255//2, size=(255-127)//16+1+8)
#     a = 1

