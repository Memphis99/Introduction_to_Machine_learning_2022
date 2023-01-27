import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from utils import get_triplet_dataset, classificator_model, get_features

def main():
    features = get_features()
    print("\n Getting training files")
    train_vec, train_lab = get_triplet_dataset('train_triplets.txt', features, lab=True)
    print("\n Getting testing files")
    test_vec = get_triplet_dataset('test_triplets.txt', features, lab=False)

    trained_model_path = 'resnet_backbone_shuffle'
    print("\n Verifying if the model exists in : ", trained_model_path)
    if not os.path.exists(trained_model_path):
        print(f"\n Model do not exists in { trained_model_path } training")
        model = classificator_model(train_vec)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], )

        resnet = True
        if resnet:
            epochs = 50
        else:
            epochs = 38

        look_for_valid = False
        if look_for_valid:
            model.fit(x=train_vec, y=train_lab, epochs=epochs, batch_size=1, validation_split=0.2, shuffle=1)
        else:
            model.fit(x=train_vec, y=train_lab, epochs=epochs, shuffle=True)
        print("\n Finished training")
        # best accuracy 0.71
        # save trained model
        print("\n Saving model in : ", trained_model_path)
        model.save(trained_model_path)
    else:
        print("\n Your model exists in path : ", trained_model_path)
        model = keras.models.load_model(trained_model_path)

    print("\n Predicting")
    out = model.predict(test_vec)

    print("\n Creating output file")
    postprocessed_out = np.where(out < 0.5, 0, 1)
    np.savetxt("submission.txt", postprocessed_out, fmt='%d')

if __name__ == '__main__':
    main()
