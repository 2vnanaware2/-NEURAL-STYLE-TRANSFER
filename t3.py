# step 1
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

#step 2
from google.colab import files
uploaded = files.upload()

#step 3
def load_image(image_path, max_dim=512):
    img = PIL.Image.open(image_path)
    img.thumbnail((max_dim, max_dim))
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.0
    return img

content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

# Show both images
plt.subplot(1, 2, 1)
plt.imshow(tf.squeeze(content_image))
plt.title("Content Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(tf.squeeze(style_image))
plt.title("Style Image")
plt.axis('off')
plt.show()

#step 4
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')



#step 5
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Show stylized result
plt.imshow(tf.squeeze(stylized_image))
plt.title("Stylized Output")
plt.axis('off')
plt.show()
