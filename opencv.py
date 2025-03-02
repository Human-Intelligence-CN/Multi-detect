import cv2
import os


def extract_frames(video_path, output_dir, frame_interval):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    image_index = 0

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        # 如果帧读取失败，则结束循环
        if not ret:
            break

        # 根据frame_interval决定是否保存当前帧
        if frame_count % frame_interval == 0:
            # 构造图像文件名
            image_filename = os.path.join(output_dir, f"frame_{image_index:04d}.jpg")

            # 保存当前帧为图像文件
            cv2.imwrite(image_filename, frame)

            # 更新图像索引
            image_index += 1

        # 更新帧计数
        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"Extracted {image_index} frames from {video_path}")


# 示例用法
video_path = './datasets/123.mp4'
output_dir = './datasets/images/train'
frame_interval = 40  # 例如，每30帧保存一帧

extract_frames(video_path, output_dir, frame_interval)