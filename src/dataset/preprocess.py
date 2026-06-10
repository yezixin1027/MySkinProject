"""医学图像预处理流水线：DullRazor 去毛发 + 灰度世界校色 + CLAHE 增强"""

import cv2
import numpy as np


class MedicalImageProcessor:
    """皮肤镜图像专用预处理器"""

    def __init__(self, kernel_size=17):
        # 形态学核：用于检测毛发
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def remove_hair(self, image):
        """DullRazor 算法：刮除皮肤毛发"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
        return dst

    def apply_clahe(self, image):
        """CLAHE：自适应直方图均衡化，增强局部对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def gray_world_balance(self, image):
        """颜色恒常性：修正光照偏差，使病灶颜色更真实"""
        b, g, r = cv2.split(image)
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        k = (b_avg + g_avg + r_avg) / 3
        b = cv2.multiply(b, k / b_avg)
        g = cv2.multiply(g, k / g_avg)
        r = cv2.multiply(r, k / r_avg)
        return cv2.merge([b, g, r])

    def full_process(self, image):
        """一键预处理流水线：去毛 → 校色 → 增强"""
        img = self.remove_hair(image)
        img = self.gray_world_balance(img)
        img = self.apply_clahe(img)
        return img

    def process_val(self, image):
        """验证/测试模式：仅去毛 + 校色，不做 CLAHE 增强"""
        img = self.remove_hair(image)
        img = self.gray_world_balance(img)
        return img
