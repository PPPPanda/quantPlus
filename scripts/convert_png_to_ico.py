"""
PNG 转 ICO 格式脚本

将 enhanced_cw.png 转换为 enhanced_cw.ico：
1. 将灰色背景（RGB值接近）转为透明
2. 裁剪到内容区域
3. 保持图标内容最大化
4. 生成多尺寸高质量 ICO
"""

from pathlib import Path
from PIL import Image
import struct
import io
import numpy as np


def remove_gray_background_and_crop(img: Image.Image, gray_tolerance: int = 15) -> Image.Image:
    """
    去除灰色背景并裁剪到内容区域。

    灰色定义：R、G、B 三个通道值的差异都小于 tolerance。

    Args:
        img: 输入图片 (RGBA)
        gray_tolerance: 灰色判定容差

    Returns:
        处理后的图片
    """
    data = np.array(img)
    r, g, b, a = data[:, :, 0].astype(int), data[:, :, 1].astype(int), data[:, :, 2].astype(int), data[:, :, 3]

    # 灰色判定：RGB 三通道值接近
    is_gray = (
        (np.abs(r - g) < gray_tolerance) &
        (np.abs(g - b) < gray_tolerance) &
        (np.abs(r - b) < gray_tolerance)
    )

    gray_count = np.sum(is_gray)
    total_count = data.shape[0] * data.shape[1]
    print(f"  灰色像素: {gray_count} ({gray_count/total_count*100:.1f}%)")
    print(f"  彩色像素: {total_count - gray_count} ({(total_count - gray_count)/total_count*100:.1f}%)")

    # 将灰色区域设为透明
    new_data = data.copy()
    new_data[:, :, 3] = np.where(is_gray, 0, 255)

    new_img = Image.fromarray(new_data, mode='RGBA')

    # 裁剪到非透明区域
    bbox = new_img.getbbox()
    if bbox:
        new_img = new_img.crop(bbox)
        print(f"  裁剪区域: {bbox}")
        print(f"  裁剪后尺寸: {new_img.width}x{new_img.height}")

    return new_img


def make_square_with_padding(img: Image.Image, padding_ratio: float = 0.05) -> Image.Image:
    """
    将图片转换为正方形，添加少量边距。
    """
    max_dim = max(img.width, img.height)
    padding = int(max_dim * padding_ratio)
    target_size = max_dim + 2 * padding

    result = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    x = (target_size - img.width) // 2
    y = (target_size - img.height) // 2
    result.paste(img, (x, y), img)

    return result


def create_ico_file(images: list[Image.Image], output_path: Path) -> None:
    """
    手动构建 ICO 文件（PNG 格式存储）。
    """
    num_images = len(images)

    image_data_list = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        image_data_list.append(buf.getvalue())

    header_size = 6 + 16 * num_images
    current_offset = header_size

    icondir = struct.pack('<HHH', 0, 1, num_images)

    entries = []
    for i, img in enumerate(images):
        width = img.width if img.width < 256 else 0
        height = img.height if img.height < 256 else 0
        data_size = len(image_data_list[i])
        entry = struct.pack('<BBBBHHII',
                            width, height, 0, 0, 1, 32,
                            data_size, current_offset)
        entries.append(entry)
        current_offset += data_size

    with open(output_path, 'wb') as f:
        f.write(icondir)
        for entry in entries:
            f.write(entry)
        for data in image_data_list:
            f.write(data)


def convert_png_to_ico(png_path: Path, ico_path: Path) -> None:
    """
    将 PNG 图片转换为 ICO 格式。
    """
    sizes = [16, 24, 32, 48, 64, 128, 256]

    print(f"读取图片: {png_path}")
    img = Image.open(png_path)
    print(f"原始尺寸: {img.width}x{img.height}")

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # 去除灰色背景并裁剪
    print("处理背景...")
    img = remove_gray_background_and_crop(img)

    # 转为正方形
    img = make_square_with_padding(img, padding_ratio=0.05)
    print(f"正方形尺寸: {img.width}x{img.height}")

    # 生成各尺寸
    icon_images = []
    for size in sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        icon_images.append(resized)
        print(f"  生成 {size}x{size}")

    # 构建 ICO
    create_ico_file(icon_images, ico_path)

    file_size = ico_path.stat().st_size
    print(f"\n转换成功: {ico_path}")
    print(f"ICO 文件大小: {file_size / 1024:.1f} KB")

    # 保存预览
    preview_path = ico_path.with_suffix('.preview.png')
    icon_images[-1].save(preview_path)
    print(f"预览图: {preview_path}")


def main() -> None:
    base_dir = Path(__file__).parent.parent
    png_path = base_dir / "src" / "qp" / "apps" / "enhanced_chart" / "ui" / "enhanced_cw.png"
    ico_path = base_dir / "src" / "qp" / "apps" / "enhanced_chart" / "ui" / "enhanced_cw.ico"

    if not png_path.exists():
        print(f"错误: PNG 文件不存在: {png_path}")
        return

    convert_png_to_ico(png_path, ico_path)


if __name__ == "__main__":
    main()
