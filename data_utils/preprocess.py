import cv2
import numpy as np

class MedicalImageProcessor:
    def __init__(self, kernel_size=17):
        # 形态学核：用于检测毛发
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def remove_hair(self, image):
        """DullRazor 算法：刮除皮肤毛发"""
        # 1. 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 黑帽运算：提取黑色线状毛发
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        # 3. 阈值处理
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        # 4. 修复 (Inpainting)
        dst = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
        return dst

    def apply_clahe(self, image):
        """CLAHE：自适应直方图均衡化，增强对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def gray_world_balance(self, image):
        """颜色恒常性：修正光照偏差，使病灶颜色更真实"""
        # 计算各通道均值
        b, g, r = cv2.split(image)
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        # 计算全局均值
        k = (b_avg + g_avg + r_avg) / 3
        # 调整各通道
        b = cv2.multiply(b, k / b_avg)
        g = cv2.multiply(g, k / g_avg)
        r = cv2.multiply(r, k / r_avg)
        return cv2.merge([b, g, r])

    def full_process(self, image):
        """一键处理流水线"""
        img = self.remove_hair(image)      # 第一步：刮毛
        img = self.gray_world_balance(img) # 第二步：校色
        img = self.apply_clahe(img)        # 第三步：增强
        return img