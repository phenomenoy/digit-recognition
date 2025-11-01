# predict.py
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 加载模型
print("正在加载模型...")
model = tf.keras.models.load_model('mnist_digit_model.keras')
print("模型加载完成！")

def predict_digit(image_path):
    img = Image.open(image_path).convert('L')
    print(f"原始尺寸: {img.size}")

    # 反转 + 智能裁剪 + 居中缩放
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 填充成正方形
    w, h = img.size
    size = max(w, h, 20)  # 至少20
    bg = Image.new('L', (size, size), 0)
    bg.paste(img, ((size-w)//2, (size-h)//2))
    img = bg

    # 缩放到20x20 → 填充到28x28（模拟手写体）
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    final = Image.new('L', (28, 28), 0)
    final.paste(img, (4, 4))
    img = final

    # 转为数组
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = arr.reshape(1, 28, 28, 1).astype('float32') / 255

    # 预测
    pred = model.predict(arr, verbose=0)
    digit = int(pred.argmax())
    conf = float(pred.max())

    return digit, conf

# 测试
if __name__ == "__main__":
    path = r'E:\yolo\src\static\image\3.jpg'
    digit, conf = predict_digit(path)
    print(f"预测结果: {digit}  (置信度: {conf:.2%})")