from MODEL.model import CapsuleModel
from MODEL.capslayer import CapsLayer
import tensorflow as tf
import numpy as np

model = CapsuleModel(input_shape=(28, 28, 1))
dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)

# 保存为 TF SavedModel 格式
model.model.save("MODEL/capsule_model_tf", save_format="tf")

# 重新加载
loaded_model = tf.keras.models.load_model("MODEL/capsule_model_tf", custom_objects={'CapsLayer': CapsLayer})
print("✅ TF model loaded successfully.")

# 测试推理
output = loaded_model.predict(dummy_input)
print("✅ Output shape:", output.shape)
