from demo_function import *
from keras.models import load_model

# Load training and testing datasets.
ROOT_PATH = "data"
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")
images, labels = load_data(train_data_dir)
print("Unique Labels:%d\nTotal Images:%d" % (len(set(labels)), len(images)))
# 展示每种标签的第一个图像
display_images_and_labels(images, labels)
images_a,labels_a = train_dataset(images,labels)
# 打乱数据，分成训练集和验证集
train_x,train_y,val_x,val_y = shuffle_data(images_a,labels_a)
# 构建神经网络
# history = build_model(train_x,train_y,val_x,val_y)
# 画出损失图像
# plot_training(history=history)
# 载入训练好的模型
model = load_model("model.h5")
# 载入测试集
test_x,test_y,test_labels_a = test_dataset(test_data_dir)

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict(test_x)
predicted = np.argmax(predicted, 1)
print(predicted.shape)

display_prediction(test_x,test_labels_a,predicted)
plt.show()