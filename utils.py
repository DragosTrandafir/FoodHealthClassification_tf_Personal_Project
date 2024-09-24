import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_models():
    
    tf.random.set_seed(20)
    
    model_1 = Sequential(
        [
            Dense(25, activation = 'relu'),
            Dense(15, activation = 'relu'),
            Dense(1, activation = 'linear')
        ],
        name='model_1'
    )

    model_2 = Sequential(
        [
            Dense(20, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(20, activation = 'relu'),
            Dense(1, activation = 'linear')
        ],
        name='model_2'
    )

    model_3 = Sequential(
        [
            Dense(32, activation = 'relu'),
            Dense(16, activation = 'relu'),
            Dense(8, activation = 'relu'),
            Dense(4, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(1, activation = 'linear')
        ],
        name='model_3'
    )
    model_4 = Sequential(
        [
            Dense(20, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_4'
    )
    model_5 = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(20, activation='relu'),
            Dense(14, activation='relu'),
            Dense(6, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_5'
    )
    model_list = [model_1, model_2, model_3,model_4,model_5]
    
    return model_list


