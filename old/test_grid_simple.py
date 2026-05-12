from PIL import Image, ImageDraw

def create_grid_image():
    # 创建一个 1024x1024 的白色画布
    img = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(img)

    # 每个格子大小 512x512
    cell_size = 512

    # 绘制四个格子边框
    for row in range(2):
        for col in range(2):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    # 保存图片
    img.save("grid_1024.png")
    print("生成完成：grid_1024.png")

if __name__ == "__main__":
    create_grid_image()
