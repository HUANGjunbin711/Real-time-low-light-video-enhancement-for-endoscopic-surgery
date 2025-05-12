import cv2
import os

# 读取图像路径
path = r"D:\Program Files\LLVE_Endoscopic\Endovis17\Endovis17\eval15\low"
img_files = sorted([f for f in os.listdir(path) if f.endswith((".jpg", ".png"))])
first_img = cv2.imread(os.path.join(path, img_files[0]))
h, w = first_img.shape[:2]
size = (w, h)

# 初始化写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowrite = cv2.VideoWriter(r"D:\Program Files\LLVE_Endoscopic\Endovis17\Endovis17\eval15\test.mp4", fourcc, 25, size)

for fn in img_files:
    filename = os.path.join(path, fn)
    img = cv2.imread(filename)
    if img is None:
        print(filename + "为空!")
        continue
    img_resized = cv2.resize(img, size)  # 保证尺寸一致
    videowrite.write(img_resized)

videowrite.release()
print('视频已保存完毕！')
