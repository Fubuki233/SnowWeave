from core.image_grid_generator import ImageGridGenerator

generator = ImageGridGenerator()

result = generator.save_result(
    image_path="D:\\SnowGlobe\\SnowWeave\\test.png",
    image_type="plant",
    output_dir="output"
)

print("生成的文件:", result['files'])