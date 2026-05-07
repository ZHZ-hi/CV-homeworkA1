import numpy as np
from PIL import Image
import os

def rgb_to_hsv(rgb_image):
    rgb = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val
    
    h = np.zeros_like(max_val)
    s = np.zeros_like(max_val)
    v = max_val.copy()
    
    s = np.where(max_val != 0, diff / max_val, 0)
    
    mask_r = (max_val == r)
    mask_g = (max_val == g) & ~mask_r
    mask_b = (max_val == b) & ~mask_r & ~mask_g
    
    safe_diff = np.where(diff == 0, 1, diff)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / safe_diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / safe_diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / safe_diff[mask_b] + 4
    
    h = h / 6.0
    h = np.clip(h, 0, 1)
    
    hsv = np.stack([h, s, v], axis=2)
    return (hsv * 255).astype(np.uint8)

def rgb_to_hls(rgb_image):
    rgb = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    l = (max_val + min_val) / 2.0
    
    diff = max_val - min_val
    s = np.where(diff != 0, diff / (1 - np.abs(2 * l - 1)), 0)
    
    h = np.zeros_like(max_val)
    mask_r = (max_val == r)
    mask_g = (max_val == g) & ~mask_r
    mask_b = (max_val == b) & ~mask_r & ~mask_g
    
    safe_diff = np.where(diff == 0, 1, diff)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / safe_diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / safe_diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / safe_diff[mask_b] + 4
    
    h = h / 6.0
    h = np.clip(h, 0, 1)
    
    hls = np.stack([h, l, s], axis=2)
    return (hls * 255).astype(np.uint8)

def rgb_to_yuv(rgb_image):
    rgb = rgb_image.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128
    
    yuv = np.stack([y, u, v], axis=2).astype(np.uint8)
    return yuv

def rgb_to_ycrcb(rgb_image):
    rgb = rgb_image.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    
    ycrcb = np.stack([y, cb, cr], axis=2).astype(np.uint8)
    return ycrcb

def merge_rgb(r, g, b):
    return np.stack([r, g, b], axis=2)

def display_channels(image_path, output_dir):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    
    os.makedirs(output_dir, exist_ok=True)
    
    name = os.path.splitext(os.path.basename(image_path))[0]
    
    Image.fromarray(img).save(f"{output_dir}/{name}_original.png")
    
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    zeros = np.zeros_like(r)
    Image.fromarray(r).save(f"{output_dir}/{name}_R.png")
    Image.fromarray(g).save(f"{output_dir}/{name}_G.png")
    Image.fromarray(b).save(f"{output_dir}/{name}_B.png")
    Image.fromarray(merge_rgb(r, zeros, zeros)).save(f"{output_dir}/{name}_R_channel.png")
    Image.fromarray(merge_rgb(zeros, g, zeros)).save(f"{output_dir}/{name}_G_channel.png")
    Image.fromarray(merge_rgb(zeros, zeros, b)).save(f"{output_dir}/{name}_B_channel.png")
    
    hsv = rgb_to_hsv(img)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    Image.fromarray(h).save(f"{output_dir}/{name}_HSV_H.png")
    Image.fromarray(s).save(f"{output_dir}/{name}_HSV_S.png")
    Image.fromarray(v).save(f"{output_dir}/{name}_HSV_V.png")
    Image.fromarray(merge_rgb(h, h, h)).save(f"{output_dir}/{name}_HSV_H_channel.png")
    Image.fromarray(merge_rgb(s, s, s)).save(f"{output_dir}/{name}_HSV_S_channel.png")
    Image.fromarray(merge_rgb(v, v, v)).save(f"{output_dir}/{name}_HSV_V_channel.png")
    
    hls = rgb_to_hls(img)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    Image.fromarray(h).save(f"{output_dir}/{name}_HLS_H.png")
    Image.fromarray(l).save(f"{output_dir}/{name}_HLS_L.png")
    Image.fromarray(s).save(f"{output_dir}/{name}_HLS_S.png")
    Image.fromarray(merge_rgb(h, h, h)).save(f"{output_dir}/{name}_HLS_H_channel.png")
    Image.fromarray(merge_rgb(l, l, l)).save(f"{output_dir}/{name}_HLS_L_channel.png")
    Image.fromarray(merge_rgb(s, s, s)).save(f"{output_dir}/{name}_HLS_S_channel.png")
    
    yuv = rgb_to_yuv(img)
    y, u, v = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    Image.fromarray(y).save(f"{output_dir}/{name}_YUV_Y.png")
    Image.fromarray(u).save(f"{output_dir}/{name}_YUV_U.png")
    Image.fromarray(v).save(f"{output_dir}/{name}_YUV_V.png")
    Image.fromarray(merge_rgb(y, y, y)).save(f"{output_dir}/{name}_YUV_Y_channel.png")
    Image.fromarray(merge_rgb(u, u, u)).save(f"{output_dir}/{name}_YUV_U_channel.png")
    Image.fromarray(merge_rgb(v, v, v)).save(f"{output_dir}/{name}_YUV_V_channel.png")
    
    ycrcb = rgb_to_ycrcb(img)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    Image.fromarray(y).save(f"{output_dir}/{name}_YCrCb_Y.png")
    Image.fromarray(cr).save(f"{output_dir}/{name}_YCrCb_Cr.png")
    Image.fromarray(cb).save(f"{output_dir}/{name}_YCrCb_Cb.png")
    Image.fromarray(merge_rgb(y, y, y)).save(f"{output_dir}/{name}_YCrCb_Y_channel.png")
    Image.fromarray(merge_rgb(cr, cr, cr)).save(f"{output_dir}/{name}_YCrCb_Cr_channel.png")
    Image.fromarray(merge_rgb(cb, cb, cb)).save(f"{output_dir}/{name}_YCrCb_Cb_channel.png")
    
    print(f"处理完成，结果保存在: {output_dir}")
    print("生成的文件:")
    print("  RGB: R.png, G.png, B.png")
    print("  HSV: H.png, S.png, V.png")
    print("  HLS: H.png, L.png, S.png")
    print("  YUV: Y.png, U.png, V.png")
    print("  YCrCb: Y.png, Cr.png, Cb.png")

def nearest_neighbor_interpolate(image, new_width, new_height):
    src_height, src_width = image.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    scale_y = src_height / new_height
    scale_x = src_width / new_width
    
    for y in range(new_height):
        for x in range(new_width):
            src_y = int((y + 0.5) * scale_y - 0.5)
            src_x = int((x + 0.5) * scale_x - 0.5)
            src_y = np.clip(src_y, 0, src_height - 1)
            src_x = np.clip(src_x, 0, src_width - 1)
            result[y, x] = image[src_y, src_x]
    
    return result

def bilinear_interpolate(image, new_width, new_height):
    src_height, src_width = image.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=np.float32)
    
    scale_y = (src_height - 1) / (new_height - 1) if new_height > 1 else 0
    scale_x = (src_width - 1) / (new_width - 1) if new_width > 1 else 0
    
    for y in range(new_height):
        for x in range(new_width):
            src_y = y * scale_y
            src_x = x * scale_x
            
            y0 = int(src_y)
            x0 = int(src_x)
            y1 = min(y0 + 1, src_height - 1)
            x1 = min(x0 + 1, src_width - 1)
            
            fy = src_y - y0
            fx = src_x - x0
            
            p00 = image[y0, x0].astype(np.float32)
            p10 = image[y0, x1].astype(np.float32)
            p01 = image[y1, x0].astype(np.float32)
            p11 = image[y1, x1].astype(np.float32)
            
            result[y, x] = (1 - fx) * (1 - fy) * p00 + \
                           fx * (1 - fy) * p10 + \
                           (1 - fx) * fy * p01 + \
                           fx * fy * p11
    
    return result.astype(np.uint8)

def scale_image(image, scale_factor, method):
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    if method == "nearest":
        return nearest_neighbor_interpolate(image, new_width, new_height)
    else:
        return bilinear_interpolate(image, new_width, new_height)

def resize_image(image, new_width, new_height, method):
    if method == "nearest":
        return nearest_neighbor_interpolate(image, new_width, new_height)
    else:
        return bilinear_interpolate(image, new_width, new_height)

def rotate_image(image, angle, method):
    h, w = image.shape[:2]
    center_y, center_x = h / 2, w / 2
    
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    new_h = int(abs(-sin_a * w) + abs(cos_a * h)) + 2
    new_w = int(abs(cos_a * w) + abs(-sin_a * h)) + 2
    
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_center_y, new_center_x = new_h / 2, new_w / 2
    
    for y in range(new_h):
        for x in range(new_w):
            dx = x - new_center_x
            dy = y - new_center_y
            
            src_x = dx * cos_a + dy * sin_a + center_x
            src_y = -dx * sin_a + dy * cos_a + center_y
            
            src_x = np.clip(src_x, 0, w - 1)
            src_y = np.clip(src_y, 0, h - 1)
            
            if method == "nearest":
                result[y, x] = image[int(src_y), int(src_x)]
            else:
                x0, y0 = int(src_x), int(src_y)
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)
                
                fx = src_x - x0
                fy = src_y - y0
                
                p00 = image[y0, x0].astype(np.float32)
                p10 = image[y0, x1].astype(np.float32)
                p01 = image[y1, x0].astype(np.float32)
                p11 = image[y1, x1].astype(np.float32)
                
                result[y, x] = ((1 - fx) * (1 - fy) * p00 + 
                               fx * (1 - fy) * p10 + 
                               (1 - fx) * fy * p01 + 
                               fx * fy * p11).astype(np.uint8)
    
    return result

def interpolate_main(image_path, operation, params, method):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    
    output_dir = "output_interpolation"
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    
    result = None
    output_name = ""
    
    if operation == "scale":
        factor = params["factor"]
        result = scale_image(img, factor, method)
        output_name = f"{name}_{method}_scale_{factor}x.png"
        
    elif operation == "resize":
        width = params["width"]
        height = params["height"]
        result = resize_image(img, width, height, method)
        output_name = f"{name}_{method}_resize_{width}x{height}.png"
        
    elif operation == "rotate":
        angle = params["angle"]
        result = rotate_image(img, angle, method)
        output_name = f"{name}_{method}_rotate_{angle}deg.png"
    
    Image.fromarray(result).save(f"{output_dir}/{output_name}")
    print(f"操作: {operation} ({method})")
    print(f"参数: {params}")
    print(f"输出: {output_dir}/{output_name}")
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interpolate":
            image_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
            operation = sys.argv[3] if len(sys.argv) > 3 else "scale"
            method = sys.argv[4] if len(sys.argv) > 4 else "bilinear"
            
            if operation == "scale":
                factor = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
                params = {"factor": factor}
            elif operation == "resize":
                width = int(sys.argv[5]) if len(sys.argv) > 5 else 400
                height = int(sys.argv[6]) if len(sys.argv) > 6 else 300
                params = {"width": width, "height": height}
            elif operation == "rotate":
                angle = float(sys.argv[5]) if len(sys.argv) > 5 else 45
                params = {"angle": angle}
            else:
                params = {}
            
            interpolate_main(image_path, operation, params, method)
        else:
            image_path = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_channels"
            display_channels(image_path, output_dir)
    else:
        image_path = "test.jpg"
        output_dir = "output_channels"
        display_channels(image_path, output_dir)