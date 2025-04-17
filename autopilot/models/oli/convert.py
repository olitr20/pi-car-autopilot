import tensorflow as tf

new_model = tf.keras.models.load_model('ResNet34_Aug_Bright_fc1000.keras')

# Check its architecture
new_model.summary()

#Save the model into h5 format
new_model.save('ResNet34_Aug_Bright_fc1000.h5')
