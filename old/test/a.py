import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import pipeline
image_path = r"D:\\SnowGlobe\\SnowWeave\\out\\crystal-wolf-all\\raw-1.jpg"
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
pillow_image = pipe(image_path) # applies mask on input and returns a pillow image
pillow_mask.save("mask.png")
pillow_image.save("result.png")