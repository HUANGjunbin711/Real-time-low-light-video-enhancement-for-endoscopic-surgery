import cv2
import numpy as np
from collections import deque
import os
import math

# ---------- 参数设置 ----------
LOG_STRENGTH = math.e - 1     # log增强强度（越大越亮），推荐范围：0.6 ~ 1.5
LOW_LIGHT_THRESHOLD = 50      # 低光检测亮度阈值
MOTION_REJECT_THRESHOLD = 40  # 当运动强度高于此阈值时，放弃融合
MOTION_POWER = 0.5  # [推荐范围 0.3 ~ 0.7] 越小 → 越弱化运动评分影响
SAVE_OUTPUT = True
SAVE_DEBUG_IMAGES = True
OUTPUT_PATH = "enhanced_output_0418.mp4"
FRAME_SIZE = (320, 256)
DEBUG_FOLDER = "debug_output_0418"

# ---------- 创建调试输出文件夹 ----------
if SAVE_DEBUG_IMAGES and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)


# ---------- 判断是否为低光帧 ----------
def is_low_light(image, threshold=LOW_LIGHT_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold


# ---------- Log增强 ----------
def log_correction(image, strength=LOG_STRENGTH):
    img_float = image.astype(np.float32) / 255.0
    img_log = np.log1p(img_float * strength)
    img_log = img_log / np.log1p(strength)
    img_log = (img_log * 255).clip(0, 255).astype(np.uint8)
    return img_log


# ---------- LIME增强 ----------
def lime_enhancement_refined(image_raw, image, beta=0.5, eps=1e-4, r=15):
    """
    LIME增强（包含光照图平滑），基于 Guo et al. 2017
    :param image: 输入RGB图像，uint8
    :param beta: 增强指数，越小越亮，推荐0.6~0.9
    :param eps: 引导滤波epsilon，越小越保边
    :param r: 引导滤波窗口大小
    :return: 增强图像
    """

    # 1. 归一化图像
    img = image.astype(np.float32) / 255.0
    image_raw = image_raw.astype(np.float32) / 255.0

    # 2. 初始光照图（最大通道）
    T_init = np.max(image_raw, axis=2)

    # 3. 使用引导滤波对 T_init 平滑，输入图像自己作为引导图
    # Convert guide to gray for filtering
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    T_refined = cv2.ximgproc.guidedFilter(guide=guide, src=T_init, radius=r, eps=eps)

    # 4. 限制最小亮度，防止除零
    T_refined = np.clip(T_refined, 0.01, 1.0)

    # 5. 增强反射率图像（逐通道）
    enhanced = img / (T_refined[:, :, np.newaxis] ** beta)
    enhanced = np.clip(enhanced, 0, 1.0)

    return (enhanced * 255).astype(np.uint8)


# ---------- 光流对齐 ----------
def align_frame(ref_frame, target_frame):
    gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_target, gray_ref, None,
        pyr_scale=0.3, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0
    )

    h, w = gray_ref.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    remap_x = (grid_x + flow[..., 0]).astype(np.float32)
    remap_y = (grid_y + flow[..., 1]).astype(np.float32)

    aligned = cv2.remap(target_frame, remap_x, remap_y,
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return aligned, flow  # 增加返回 flow



def compute_motion_strength(curr, other):
    diff = cv2.absdiff(curr, other)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def detail_score_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)  # shape: [H, W]
    return grad


def motion_score_map(curr, other, flow=None, use_confidence=True, confidence_tau=20.0):
    """
    计算运动评分图：越稳定越高，可用已有光流加速。
    flow 可来自 align_frame，若尺寸不同则自动缩放。
    """
    diff = cv2.absdiff(curr, other)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion_score = 1.0 / (gray_diff + 1.0)

    if use_confidence and flow is not None:
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        other_gray = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)

        h, w = curr_gray.shape[:2]

        # 若 flow 大小和图像不一致，自动缩放（线性插值）
        if flow.shape[:2] != (h, w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow = flow * (w / flow.shape[1], h / flow.shape[0])  # 缩放矢量幅度

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped_other = cv2.remap(other_gray, map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        warp_error = cv2.absdiff(curr_gray, warped_other).astype(np.float32)
        confidence = np.exp(-warp_error / confidence_tau)

        motion_score *= confidence

    return motion_score


def local_noise_map(absdiff_gray):
    return cv2.GaussianBlur(absdiff_gray, (5, 5), 0)


def unsharp_mask(image, strength=1.2, kernel_size=(5, 5), sigma=1.0):
    """
    Unsharp Mask 锐化图像以增强细节。
    strength：锐化强度，建议 0.5 ~ 1.5
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def restore_color_contrast_lab(image, clip_limit=1.0, tile_grid_size=(5, 5)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------- 智能时序融合 ----------(后续加入运动检测补偿机制，还有光照不稳定怎么解决)

def smart_temporal_fusion(curr, aligned_prev, aligned_next, flow_prev, flow_next, frame_idx, scale=0.5):
    # === Frame-level motion strength detection with fallback ===
    motion_strength_prev = compute_motion_strength(curr, aligned_prev)
    motion_strength_next = compute_motion_strength(curr, aligned_next)

    if motion_strength_prev > MOTION_REJECT_THRESHOLD and motion_strength_next > MOTION_REJECT_THRESHOLD:
        # 两侧运动都太剧烈，放弃融合，直接用当前帧增强
        if SAVE_DEBUG_IMAGES:
            base = os.path.join(DEBUG_FOLDER, f"frame_{frame_idx:04d}_")
            cv2.imwrite(base + "fusion_skipped.png", curr)
        print(
            f"[Frame {frame_idx}] Skip fusion (both motions high): prev={motion_strength_prev:.2f}, next={motion_strength_next:.2f}")
        output = curr.copy()
        output = unsharp_mask(output, strength=0.8)
        output = restore_color_contrast_lab(output, clip_limit=1.0, tile_grid_size=(8, 8))
        return output

    elif motion_strength_prev > MOTION_REJECT_THRESHOLD:
        # 仅前帧运动太大，用当前 + 后帧进行融合
        print(f"[Frame {frame_idx}] Only prev motion high ({motion_strength_prev:.2f}), use curr + next")
        aligned_prev = curr.copy()  # 将前帧替换为当前帧（等效于去掉）

    elif motion_strength_next > MOTION_REJECT_THRESHOLD:
        # 仅后帧运动太大，用当前 + 前帧进行融合
        print(f"[Frame {frame_idx}] Only next motion high ({motion_strength_next:.2f}), use curr + prev")
        aligned_next = curr.copy()

    h, w = curr.shape[:2]

    def resize_if_needed(img):
        if scale == 1.0:
            return img
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Resize inputs if needed
    curr_s = resize_if_needed(curr)
    prev_s = resize_if_needed(aligned_prev)
    next_s = resize_if_needed(aligned_next)

    # Weight components at possibly smaller scale
    map_prev = detail_score_map(prev_s)
    map_curr = detail_score_map(curr_s)
    map_next = detail_score_map(next_s)

    motion_prev = motion_score_map(curr_s, prev_s, flow=flow_prev) ** MOTION_POWER
    motion_curr = np.ones_like(motion_prev)
    motion_next = motion_score_map(curr_s, next_s, flow=flow_next) ** MOTION_POWER

    def local_noise(gray): return cv2.GaussianBlur(cv2.absdiff(gray, cv2.medianBlur(gray, 5)), (5, 5), 0)

    noise_prev = 1.0 / (local_noise(cv2.cvtColor(prev_s, cv2.COLOR_BGR2GRAY)).astype(np.float32) + 10)
    noise_curr = 1.0 / (local_noise(cv2.cvtColor(curr_s, cv2.COLOR_BGR2GRAY)).astype(np.float32) + 10)
    noise_next = 1.0 / (local_noise(cv2.cvtColor(next_s, cv2.COLOR_BGR2GRAY)).astype(np.float32) + 10)

    # Weighted maps
    w_prev = cv2.GaussianBlur(map_prev * motion_prev * noise_prev, (13, 13), 5)
    w_curr = cv2.GaussianBlur(map_curr * motion_curr * noise_curr, (13, 13), 5)
    w_next = cv2.GaussianBlur(map_next * motion_next * noise_next, (13, 13), 5)

    # Resize back if scaled
    if scale != 1.0:
        w_prev = cv2.resize(w_prev, (w, h), interpolation=cv2.INTER_LINEAR)
        w_curr = cv2.resize(w_curr, (w, h), interpolation=cv2.INTER_LINEAR)
        w_next = cv2.resize(w_next, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize weights
    total = w_prev + w_curr + w_next + 1e-6
    w_prev /= total
    w_curr /= total
    w_next = 1.0 - w_prev - w_curr

    # Weighted fusion
    fused = (aligned_prev.astype(np.float32) * w_prev[:, :, np.newaxis] +
             curr.astype(np.float32) * w_curr[:, :, np.newaxis] +
             aligned_next.astype(np.float32) * w_next[:, :, np.newaxis])
    output = np.clip(fused, 0, 255).astype(np.uint8)

    # Post-enhancement
    # output = unsharp_mask(output, strength=0.8)
    # output = restore_color_contrast_lab(output, clip_limit=1.0, tile_grid_size=(8, 8))

    if SAVE_DEBUG_IMAGES:
        base = os.path.join(DEBUG_FOLDER, f"frame_{frame_idx:04d}_")
        cv2.imwrite(base + "fused.png", output)
        cv2.imwrite(base + "w_prev.png", (w_prev * 255).astype(np.uint8))
        cv2.imwrite(base + "w_curr.png", (w_curr * 255).astype(np.uint8))
        cv2.imwrite(base + "w_next.png", (w_next * 255).astype(np.uint8))

        cv2.imwrite(base + "noise_prev.png", (noise_prev * 700).astype(np.uint8))
        cv2.imwrite(base + "noise_curr.png", (noise_curr * 700).astype(np.uint8))
        cv2.imwrite(base + "noise_next.png", (noise_next * 700).astype(np.uint8))

        cv2.imwrite(base + "output.png", output)


    return output


# ---------- 主流程 ----------
def process_video(video_path="seq.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = deque(maxlen=3)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, FRAME_SIZE)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        frame_buffer.append(frame)

        if len(frame_buffer) < 3:
            continue

        prev_raw, curr_raw, next_raw = frame_buffer

        if is_low_light(curr_raw):
            # 先对当前帧增强
            curr = log_correction(curr_raw)
            curr = lime_enhancement_refined(curr_raw, curr)

            # 对 next 帧判断是否增强
            if is_low_light(next_raw):
                next_ = log_correction(next_raw)
                next_ = lime_enhancement_refined(next_raw, next_)
            else:
                next_ = next_raw

            base = os.path.join(DEBUG_FOLDER, f"frame_{frame_idx:04d}_")
            cv2.imwrite(base + "curr.png", curr)
            cv2.imwrite(base + "next_.png", next_)


            prev = prev_raw

            # 对齐与融合
            aligned_prev, flow_prev = align_frame(curr, prev)
            aligned_next, flow_next = align_frame(curr, next_)

            cv2.imwrite(base + "aligned_prev.png", aligned_prev)
            cv2.imwrite(base + "aligned_next.png", aligned_next)

            enhanced = smart_temporal_fusion(curr, aligned_prev, aligned_next, flow_prev, flow_next, frame_idx)
        else:
            enhanced = curr_raw.copy()

        cv2.imshow("Enhanced", enhanced)
        if SAVE_OUTPUT:
            out_writer.write(enhanced)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if SAVE_OUTPUT:
        out_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 视频保存至：{OUTPUT_PATH}")
    if SAVE_DEBUG_IMAGES:
        print(f"[INFO] 调试图像保存在：{DEBUG_FOLDER}/")

# ---------- 启动 ----------
if __name__ == "__main__":
    process_video("seq.mp4")
