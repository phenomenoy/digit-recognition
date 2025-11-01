import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
from PIL import Image, ImageOps

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 添加数据增强以提升泛化能力
datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转角度
    width_shift_range=0.1,  # 水平平移
    height_shift_range=0.1, # 垂直平移
    shear_range=0.1,        # 剪切变换
    zoom_range=0.1          # 缩放
)

# 构建CNN模型（添加Dropout防止过拟合）
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))  # 添加Dropout层
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型（增加epochs到10，提升准确率）
model.fit(datagen.flow(train_images, train_labels, batch_size=64),
          epochs=10,
          validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# 保存模型
model.save('handwritten_digit_model.keras')

# 加载测试图片并进行预测（改进预处理：添加反转以匹配MNIST格式，假设用户图片可能是白底黑字）
def predict_image(image_path):
    # 加载图片
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    original_size = img.size  # 获取原始图片大小
    print(f'Original image size: {original_size}')
    
    # 反转图像颜色（MNIST是黑底白字，如果你的图片是白底黑字，这一步很重要）
    img = ImageOps.invert(img)
    
    # 调整图片大小到28x28（使用LANCZOS滤波器保持质量）
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255
    
    # 预测
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    
    return predicted_label, original_size

# 测试图片路径
test_image_path = r'E:\yolo\src\static\image\7.jpg'

# 预测并打印结果
predicted_label, original_size = predict_image(test_image_path)
print(f'Predicted label: {predicted_label}')
print(f'Original image size: {original_size}')