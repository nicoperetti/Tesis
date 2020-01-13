"""Autoencoder."""
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Concatenate


class AutoEncoder:
    """Autoencoder module."""

    def __init__(self, exp, weights_path):
        """Init method."""
        self.exp = exp
        self.weights_path = weights_path
        self.model = self.load_model()

    def load_model(self):
        """Load model."""
        encoding_dim = 200
        # Inputs
        input_x = Input(shape=(4096,), name="input_img")
        input_y = Input(shape=(300,), name="input_text")

        fc1 = Dense(300, activation='relu', name="fc1")(input_x)

        # "encoded" is the encoded representation of the input
        fc_share = Dense(300, activation='relu', name="fc_share")

        share_img = fc_share(fc1)
        share_word = fc_share(input_y)
        concat = Concatenate(name="concat")([share_img, share_word])
        encoded = Dense(encoding_dim, activation='relu', name="encoded")(concat)

        # "decoded" is the lossy reconstruction of the input
        decoded1 = Dense(4096, name="fc2")(encoded)
        decoded2 = Dense(300, name="fc3")(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model([input_x, input_y], [decoded1, decoded2])

        autoencoder.load_weights(self.weights_path)

        if self.exp == 1:
            model_img = Model(input=input_x, output=fc1)
            model_text = Model(input=input_y, output=share_word)
            model = (model_img, model_text)
        else:
            model = Model([input_x, input_y], encoded)
        return model

    def predict(self, kind, x=None, y=None):
        """Predict method."""
        if kind == "img":
            return self.model[0].predict(x.reshape(1, -1))[0]
        elif kind == "text":
            y = [t.reshape(1, -1) for t in y]
            return np.array([self.model[1].predict(tag)[0] for tag in y])
        elif kind == "img_text":
            return np.array([self.model.predict([x.reshape(1, -1), tag])[0] for tag in y])
        else:
            raise RuntimeError("Error during autoencoder prediction")
