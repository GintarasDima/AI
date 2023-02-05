# вся программа пишется в Coolab
# все модули разделены пустым комментарием


# Подключение класса для создания нейронной сети прямого распространения
from tensorflow.keras.models import Sequential
# Подключение класса для создания полносвязного слоя
from tensorflow.keras.layers import Dense, Flatten
# Подключение оптимизатора
from tensorflow.keras.optimizers import Adam
# Подключение утилит для to_categorical
from tensorflow.keras import utils
# Подключение библиотеки для загрузки изображений
from tensorflow.keras.preprocessing import image
# Подключение библиотеки для работы с массивами
import numpy as np
# Подключение модуля для работы с файлами
import os
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
from PIL import Image
# Вывод изображения в ноутбуке, а не в консоли или файле
#%matplotlib inline
#

# Загрузка датасета из облака
import gdown
gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l3/hw_pro.zip', None, quiet=True)
#

# Распаковываем архив hw_light.zip в папку hw_light
#!unzip -q hw_pro.zip
#

# Путь к директории с базой
base_dir = '/content/hw_pro'
# Создание пустого списка для загрузки изображений обучающей выборки
x_train = []
# Создание списка для меток классов
y_train = []
# Задание высоты и ширины загружаемых изображений
img_height = 20
img_width = 20
# Перебор папок в директории базы
for patch in os.listdir(base_dir):
    # Перебор файлов в папках
    for img in os.listdir(base_dir + '/' + patch):
        # Добавление в список изображений текущей картинки
        x_train.append(image.img_to_array(image.load_img(base_dir + '/' + patch + '/' + img,
                                                         target_size=(img_height, img_width),
                                                         color_mode='grayscale')))
        # Добавление в массив меток, соответствующих классам
        if patch == '0':
            y_train.append(0)
        else:
            y_train.append(1)
# Преобразование в numpy-массив загруженных изображений и меток классов
x_train = np.array(x_train)
y_train = np.array(y_train)
# Вывод размерностей
print('Размер массива x_train', x_train.shape)
print('Размер массива y_train', y_train.shape)
#

# Вывод формы данных для обучения

# Номер картинки
n = 62
# Отрисовка картинки
plt.imshow(np.squeeze(x_train[0], 2))
# Вывод n-й картинки
plt.show()
#

# Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
x_train = x_train / 255.0
# Преобразование y_train в тип float32 (числа с плавающей точкой) и нормализация
y_train = y_train / 255.0

y_train = utils.to_categorical(y_train)
x_train = utils.to_categorical(x_train)

print('Размер массива x_train', x_train.shape)
print('Размер массива y_train', y_train.shape)


CLASS_COUNT = 2
class_names = ['X', 'O']

#x_train = np.random.rand(102,20,20,2)
#y_train = np.random.rand(102,2)

model = Sequential()
# переформатируем входные данные в одномерный массив (20х20=400пикселов)
model.add(Flatten(input_shape=(20, 20, 2)))
# Добавление полносвязного слоя на 800 нейронов с relu-активацией
model.add(Dense(800, input_dim=400, activation='relu'))

# Добавление полносвязного слоя на 400 нейронов с relu-активацией
model.add(Dense(400, activation='relu'))

# Добавление полносвязного слоя с количеством нейронов по числу классов с sigmoid-активацией
model.add(Dense(CLASS_COUNT, activation='sigmoid'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=("adam"), metrics=["accuracy"])
#


#создаем модель для визуализации
utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)


model.fit(x_train,        # обучающая выборка, входные данные
          y_train,        # обучающая выборка, выходные данные
          batch_size=128, # кол-во примеров, которое обрабатывает нейронка перед одним изменением весов
          epochs=15,      # количество эпох, когда нейронка обучается на всех примерах выборки
          verbose=1)      # 0 - не визуализировать ход обучения, 1 - визуализировать
#

# проведение теста
test_loss, test_acc = model.evaluate(x_train,  y_train, verbose=2)
print('\nTest accuracy:', test_acc)

n_rec = np.random.randint(x_train.shape[0]) # выбор определения
x = x_train[n_rec]                          # выбор нужной картинки их тестовой выборки
print(x.shape)                              # проверка формы данных
# Добавление одной оси в начале, чтобы нейронка могла распознать пример
# Массив из одного примера, так как нейронка принимает именно массивы примеров (батчи) для распознавания
x = np.expand_dims(x, axis=0)
print(x.shape)                              # проверка формы данных
prediction = model.predict(x)               # распознование примера
pred = np.argmax(prediction)                # Получение и вывод индекса самого большого элемента (это значение цифры, которую распознала сеть)
print('Распознано: ')
if pred == 0:                               # проверяем условия совпадения обучения с правильным ответом
  print('O')
else:
  print('X')
print('Правильный ответ: ')
if y_train[n_rec] == 0:
  print('O')
else:
  print('X')

