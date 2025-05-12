import cv2
import os

# 输入视频路径
video_path = "enhanced_output_0418.mp4"

# 输出文件夹路径，保存提取的图像
output_folder = "output_frames_result0418"

# 创建输出文件夹，如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 读取视频的总帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 逐帧读取视频并保存为图片
frame_num = 0
while True:
    ret, frame = cap.read()

    # 如果读取到的帧为空，表示视频已经结束
    if not ret:
        break

    # 保存每一帧为图像文件，命名为 frame_001.jpg, frame_002.jpg, ...
    frame_filename = os.path.join(output_folder, f'frame_{frame_num:03d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_num += 1

# 释放视频捕获对象
cap.release()

print(f'所有帧已成功提取并保存到 "{output_folder}" 文件夹中，帧数：{frame_num}')
