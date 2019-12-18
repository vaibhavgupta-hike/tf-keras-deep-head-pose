import tensorflow as tf
import numpy as np
import utils


EPOCHS = 1

class AlexNet:

    def __init__(self, dataset, class_num, batch_size, input_size):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset = dataset
        self.model = self.__create_model()

    def __loss_angle(self, y_true, y_pred, alpha=0.5):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.keras.utils.to_categorical(bin_true, self.class_num),
                                logits=y_pred
                              )
        # MSE loss
        idx_tensor = tf.convert_to_tensor(range(self.class_num), dtype=tf.float32)
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * idx_tensor, axis=1) * 3 - 99  # Understand this better
        mse_loss = tf.keras.losses.MSE(y_true=cont_true, y_pred=pred_cont)
        # Total loss
        total_loss = classification_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))

        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)

        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)

        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])

        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses)
        return model

    def train(self, model_path, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()

        if load_weight:
            self.model.load_weights(model_path)
        else:
            self.model.fit_generator(generator=self.dataset.data_generator(test=False),             # Try using callbacks here
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    max_queue_size=10,
                                    workers=1,
                                    verbose=1)
            self.model.save(model_path)


    def test(self, save_dir):
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99
            # print(pred_cont_yaw.shape)

            self.dataset.save_test(names[0], save_dir, [pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]])

    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        # print(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 99

        return pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]
