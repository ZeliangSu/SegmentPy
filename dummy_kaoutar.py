
import tensorflow as tf
import numpy as np
from main_train import saver
#from layers import prediction
from model import model
from layers import placeholder
from PIL import Image
from proc import _stride

img = Image.open("./dummy")  # loading the image
img1 = np.asarray(img)       #transforming img object to array
batch = _stride(img1, 1, 100)

#img=np.float32(nor_data)

patch_size = 40
batch_size = 200


y_pred, train_op, X, y_true, prob_hold, merged = model(patch_size, conv_size, nb_conv, learning_rate=learning_rate)
#load weight

#def predict ():

with tf.Sess() as Sess :
    saver.restore (Sess,'./weight/{}_{}.ckpt'.format(patch_size, batch_size))
    print("model restored with success")

    #predict_op = tf.argmax(model, 1)
    #predict_op=

    # faire un aller
    P = sess.run(predict_op, feed_dict={X:batch ,y_true:batch})   # X is the pred for input ; (y_true) is the pred for the output .
    #return P





#reconstruction de l'image