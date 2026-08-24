{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Лабораторная работа № 5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Работа с нейронными сетями в Keras"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Подключаем необходимые библиотеки:\n",
    "- tensorflow - открытая библиотека для машинного обучения от Google для создания нейросетевых моделей\n",
    "- pyplot потребуется для отрисовки картинок и графиков\n",
    "- randint - генератор случайных чисел для выбора примеров из набора данных\n",
    "- numpy потребуется для обработки массивов чисел"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Кроме того, нам потребуется высокоуровневый фреймворк keras, который позволяет быстро и легко создавать нейронные сети из различных слоев"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Недостающие библиотеки устанавливаются обычным образом, при помощи pip. Обратите внимание, что в зависимости от установленной версии tensorflow, вам может потребоваться специфическая версия keras. Если возникает ошибка, можно обнавить версию tensorflow до последней или установить необходимую версию keras. Например: pip install keras==2.11.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from random import randint\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Для примера используем стандартный набор данных MNIST (Modified National Institute of Standards and Technology database), содержащий изображения цифр, написанных от руки. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Набор данных не обязательно скачивать вручную, его можно загрузить прямо из библиотеки:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mnist = tf.keras.datasets.mnist"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Заполним массивы для обучения и проверки при помощи mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Посмотрим на объем данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train.shape, y_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Массив x_train содержит 60 000 монохромных картинок 28х28 пикселей. Уменьшим "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Посмотрим на примеры изображений. Для отображения в оттенках серого используем соответствующую цветовую карту:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.axis('off')\n",
    "plt.imshow(x_test[randint(0, 10000)], cmap=plt.get_cmap('gray'))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Пришло время создать модель:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.models.Sequential()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Наша модель пока пустая и не содержит слоев с нейронами. Последовательные модели (Sequential) создаются послойно, слои могут быть созданы сразу при создании модели, или добавлены в имеющуются модель. Воспользуемся вторым способом:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))\n",
    "model.add(tf.keras.layers.Dense(128, activation='relu'))\n",
    "model.add(tf.keras.layers.Dropout(0.2))\n",
    "model.add(tf.keras.layers.Dense(128, activation='relu'))\n",
    "model.add(tf.keras.layers.Dense(10, activation='softmax'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Мы добавили в модель несколько слоёв:\n",
    "- Входной слой обычно имеет тип Input, но мы используем слой Flatten, который обычно используется после сверточных слоев. Этот слой может принять двумерный массив и преобразовать его в одномерный. Таким образом мы экономим операцию reshape для наших изображений.\n",
    "- Первый скрытый слой является полносвязанным и содержит 128 нейронов, активируемых функцией ReLu\n",
    "- Второй скрытый слой - слой регуляризации. Приведен для примера, с некоторой долей верятности в нем обрываются входные связи, что позволяет избежать переобучения\n",
    "- Третий скрытый слой аналогичен первому скрытому\n",
    "- Выходной слой содержит 10 нейронов для 10 классов - 10 цифр. Активируется функцией softmax."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Пришло время настроить нашу модель. Здесь мы задаем параметры оптимизации, функции потерь, метрики, а также (при необходимости) другие настройки."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Пришло время обучить модель. Попробуем 5 эпох. Отладочную инфомацию сохраним в объекте history:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "history = model.fit(x_train, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Сохраним массивы точности и потерь по эпохам:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loss = history.history['loss']\n",
    "train_acc = history.history['accuracy']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Построим графики обучения модели:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(train_loss, label='train_loss')\n",
    "plt.plot(train_acc, label='train_acc')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Acc-Loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Проверим модель на тестовом наборе данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.evaluate(x_test,  y_test, verbose=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Сохраним массив предсказанных значений:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result = model.predict(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Попробуем визуализировать результат нашей работы. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Количество картинок по Х и Y\n",
    "num_rows = 2\n",
    "num_cols = 4\n",
    "num_images = num_rows * num_cols\n",
    "#Создадим объект figure\n",
    "plt.figure(figsize=(2*2*num_cols, 2*num_rows))\n",
    "#А картинки нарисуем в нем субплотами\n",
    "for i in range(num_images):\n",
    "  #Случайный элемент массива\n",
    "  num=randint(0, 10000)\n",
    "  plt.subplot(num_rows, 2*num_cols, 2*i + 1)\n",
    "  #Отключить шкалы осей\n",
    "  plt.xticks([])\n",
    "  plt.yticks([])\n",
    "  #Рисуем картинку\n",
    "  plt.imshow(x_test[num], cmap=plt.get_cmap('gray'))\n",
    "  #Подпись к ней берем из массива, который вернула модель\n",
    "  plt.xlabel(np.argmax(result[num]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Дополнительно"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Сохраним модель в h5 формате и визуализируем ее при помощи библиотеки netron:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tf.keras.models.save_model(model, 'model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import netron\n",
    "netron.start('model.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Задания:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Выделите в датасете тестовый (test), тренировочный (train) и проверочный (val) наборы.\n",
    "1. Создайте модель со своей архитектурой сети. Обясните выбор количества и параметров слоев.\n",
    "2. Обучите модель на сформированном наборе данных\n",
    "3. Изменяя гиперпараметры, число эпох обучения, архитектуру модели, метрики проверки качества добейтесь максимальной точности классификации. Опишите процесс своих поисков\n",
    "4. Напишите общие выводы по работе. Подкрепите их графиками.\n",
    "6* Попробуйте кросс-валидацию, опишите ее влияние на результат обучения"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}