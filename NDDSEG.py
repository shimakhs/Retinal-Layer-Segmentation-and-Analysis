import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt, cm
import tensorflow as tf
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dropout, Input, concatenate, BatchNormalization, ReLU
import skimage
import time
import json
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import skimage
import skimage.feature
import numpy as np
import cv2
import tfa
import skimage
import scipy.signal as sig
from octreader import OctRead

##############LAYERS########################################################################

def structure_tensor(X, sigma=1.5):
    J = cv2.Sobel(cv2.UMat(X),cv2.CV_64F,1,0,ksize=3)
    C = cv2.GaussianBlur(cv2.multiply(J,J), (7,7), 1.5)
    try:
        C = C.get()
    except:
        pass
    return C


def get_descriptions(inputs):
      X = inputs[:,:,0]
      H_elems = skimage.feature.hessian_matrix(X, sigma=2.2, order='rc')
      h = skimage.feature.hessian_matrix_eigvals(H_elems)[0]
      h = np.expand_dims(h, -1)

      #
      ste = list(skimage.feature.structure_tensor(X, sigma=3.0, mode='reflect'))

      
      #ste = structure_tensor(X.numpy().astype(np.float32), 1.5)
      #ste = np.expand_dims(ste, -1) / 4
      Ste = [np.expand_dims(ste[i], -1) / 4 for i in range(3)]
      ste = np.concatenate(Ste, -1)
 
      Y = tf.cast(X*255, tf.uint8).numpy()
      Y = cv2.blur(Y, (5,5))

      cny = skimage.feature.canny(Y, sigma=2.0, low_threshold=0.2, high_threshold=0.5, mask=None, use_quantiles=True)*1.0
      cny = np.expand_dims(cny, -1)
      output = np.concatenate([h, ste, cny], -1)
      return output.astype(np.float32)
 
def image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = get_descriptions(img3d)
        results.append(np.expand_dims(rimg3d, axis=0))
    return np.concatenate(results, axis=0)
 
class DescriptorLayer(layers.Layer):
  def __init__(self):
    super(DescriptorLayer, self).__init__()
  
  def call(self, xinputs, training):
    output = tf.py_function(image_tensor_func, [xinputs], tf.float32)
    output = K.stop_gradient(output)
    output.set_shape((xinputs.shape[0], xinputs.shape[1], xinputs.shape[2], 5))
    return output
  def compute_output_shape( self, sin ) :
    return (sin[0], sin[1], sin[2], 5)

#####
class AttentionGate(layers.Layer):
  def __init__(self,n_intermediate_filters, kernel_size=1, dilate_rate=1,n_last_filter=None):
    super(AttentionGate, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(n_intermediate_filters, kernel_size, dilation_rate=dilate_rate, padding='same')
    self.conv2 = tf.keras.layers.Conv2D(n_intermediate_filters, kernel_size, dilation_rate=dilate_rate, padding='same')
    self.activation1 = tf.keras.layers.Activation("relu")
    n = n_intermediate_filters if n_last_filter == None else n_last_filter
    self.glayer = tf.keras.layers.Conv2D(n, 1, padding='same')
    self.activation2 = tf.keras.layers.Activation('sigmoid')
    self.BN = tf.keras.layers.BatchNormalization()
    
 
  def call(self, inputs, training):
 
    input1_conv = self.conv1(inputs[0])
    input2_conv = self.conv2(inputs[1])
    f = self.activation1(input1_conv+input2_conv)
    g = self.glayer(f)
    g = self.BN(g)
    h = self.activation2(g)
    return inputs[0] * h



class TransformerBlock(layers.Layer):
    def __init__(self, out_dim, embed_dim, krnsz=3, dlr=1,rate=0.1):
        super(TransformerBlock, self).__init__()
        num_heads = out_dim
        self.krnz = krnsz
        self.dlr = dlr
        self.att = AttentionGate(num_heads, krnsz, dlr)
        self.ffn = tf.keras.Sequential(
            [layers.Conv2D(embed_dim, (3,3), activation="relu", padding='same', dilation_rate=dlr),
             layers.Conv2D(num_heads, (3,3), padding='same', dilation_rate=dlr),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att([inputs, inputs])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        #out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        #return out1 + ffn_output
        
#####################################################################################################
################LOSSES###############################################################################
def Dice(y_true, y_pred):
  D = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3)) / tf.reduce_sum(y_pred + y_true, axis=(1,2,3))
  return D

def DiceLoss(y_true, y_pred):
  intersect = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
  union = tf.reduce_sum(y_true + y_pred, axis=(1,2,3)) + 0.1
  return 1 - 2 * intersect / union


cce = tf.keras.losses.CategoricalCrossentropy()
hinge = tf.keras.losses.CategoricalHinge()
kl = tf.keras.losses.KLDivergence()
#fce = tfa.losses.SigmoidFocalCrossEntropy(alpha=1, gamma=0.9)

def edge_loss(y_true, y_pred, C=10):
  ff = np.zeros((2,1,1,1))
  ff[0,:] = -1
  ff[1,:] = 1
  ff = np.concatenate([ff]*C, 2)
  ff = tf.constant(ff, dtype=tf.float32)
  y_pred1 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_pred, ff, [1,1,1,1], 'VALID'))[:,:,:,0:-1]
  y_pred1 = tf.nn.softmax(y_pred1, axis=1)
  y_true1 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_true, ff, [1,1,1,1], 'VALID'))[:,:,:,0:-1]

  E1 = 0.1*cce(tf.experimental.numpy.moveaxis(y_true1, 1, 3), tf.experimental.numpy.moveaxis(y_pred1, 1, 3))

  ff = np.zeros((2,1,1,1))
  ff[0,:] = 1
  ff[1,:] = -1
  ff = np.concatenate([ff]*C, 2)
  ff = tf.constant(ff, dtype=tf.float32)
  y_pred2 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_pred, ff, [1,1,1,1], 'VALID'))[:,:,:,1::]
  y_pred2 = tf.nn.softmax(y_pred2, axis=1)
  y_true2 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_true, ff, [1,1,1,1], 'VALID'))[:,:,:,1::]

  E2 = 0.1*cce(tf.experimental.numpy.moveaxis(y_true2, 1, 3), tf.experimental.numpy.moveaxis(y_pred2, 1, 3))


  return E1 + E2

cce = tf.keras.losses.CategoricalCrossentropy()
hinge = tf.keras.losses.CategoricalHinge()
kl = tf.keras.losses.KLDivergence()
#fce = tfa.losses.SigmoidFocalCrossEntropy(alpha=1, gamma=0.9)

def edge_loss2(y_pred, C=10):
  ff = np.zeros((2,1,1,1))
  ff[0,:] = -1
  ff[1,:] = 1
  ff = np.concatenate([ff]*C, 2)
  ff = tf.constant(ff, dtype=tf.float32)
  y_pred1 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_pred, ff, [1,1,1,1], 'VALID'))[:,:,:,1:-1]

  ff = np.zeros((2,1,1,1))
  ff[0,:] = 1
  ff[1,:] = -1
  ff = np.concatenate([ff]*C, 2)
  ff = tf.constant(ff, dtype=tf.float32)
  y_pred2 = tf.nn.relu(-tf.nn.depthwise_conv2d(y_pred, ff, [1,1,1,1], 'VALID'))[:,:,:,1:-1]


  return y_pred2 + y_pred1

def class_tversky(y_true, y_pred):
    smooth = 1
    true_pos = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    false_neg = tf.reduce_sum(y_true * (1-y_pred), axis=(1,2,3))
    false_pos = tf.reduce_sum((1-y_true)*y_pred, axis=(1,2,3))
    alpha = 0.9
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
 
def focal_tversky_loss(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.1
    return tf.reduce_sum(K.pow((1-pt_1), gamma))

#######################################################################################################
#################MODELS################################################################################
def get_model(ds=2):
    input_size = (None, None, 1)
    inputs = Input(input_size)
    conv1 = Conv2D(8, 3,padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=16*ds)(drop4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=16*ds)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=16*ds)(drop5)
    up6 = BatchNormalization()(up6)
    up6 = ReLU()(up6)
    up6 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(up6)
    up6 = BatchNormalization()(up6)
    up6 = ReLU()(up6)
    up6 = Dropout(0.5)(up6)

    dis = ReLU()(DescriptorLayer()(inputs))
            
    merge6 = concatenate([drop4,up6, dis], axis = -1)
    conv6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv2D(32, 2, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv6)
    up7 = BatchNormalization()(up7)
    up7 = ReLU()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)

    up8 = Conv2D(16, 2, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv7)
    up8 = BatchNormalization()(up8)
    up8 = ReLU()(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    up9 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv8)
    up9 = BatchNormalization()(up9)
    up9 = ReLU()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

    bg_model = tf.keras.models.Model(inputs = inputs, outputs = conv10, name='RTSN')

    bg_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), metrics = ['accuracy'])
    ####################
    input_size = (None, None, 1)
    inputs = Input(input_size)
    fg = tf.keras.layers.Lambda(lambda x: K.stop_gradient(x))(bg_model(inputs))
    fgi = tf.keras.layers.concatenate([inputs, fg], -1)
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(fgi)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    ###pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    ###pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    ###pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    drop4 = Dropout(0.5)(conv4)
    ###pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=16*ds)(drop4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=16*ds)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(drop5)
    up6 = BatchNormalization()(up6)
    up6 = ReLU()(up6)
    up6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(up6)
    up6 = BatchNormalization()(up6)
    up6 = ReLU()(up6)
    up6 = Dropout(0.5)(up6)

    dis = ReLU()(DescriptorLayer()(inputs))
    merge6 = concatenate([drop4,up6, dis, fg], axis = -1)

    merge6 = TransformerBlock(merge6.shape[-1], 64, dlr=1*ds)(merge6)

    conv6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=8*ds)(conv6)
    up7 = BatchNormalization()(up7)
    up7 = ReLU()(up7)
    up7 = Dropout(0.5)(up7)
    merge7 = concatenate([conv3,up7, fg], axis = 3)
    merge7 = TransformerBlock(merge7.shape[-1],64, dlr=1*ds)(merge7)

    conv7 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)

    up8 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=4*ds)(conv7)
    up8 = ReLU()(up8)
    up8 = Dropout(0.5)(up8)
    merge8 = concatenate([conv2,up8, fg], axis = 3)
    merge8 = TransformerBlock(merge8.shape[-1],64, dlr=1*ds)(merge8)

    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2*ds)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    up9 = Conv2D(32, 2, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv8)
    up9 = BatchNormalization()(up9)
    up9 = ReLU()(up9)
    merge9 = concatenate([conv1,up9, fg], axis = 3)
    merge9 = TransformerBlock(merge9.shape[-1],64, dlr=1*ds)(merge9)

    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=1*ds)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv10 = Conv2D(10, 1, activation = 'softmax')(conv9)

    model = tf.keras.models.Model(inputs = inputs, outputs = conv10, name='RLS')

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss=focal_tversky_loss , metrics = ['accuracy'])
    ######
    return bg_model, model
#################################################################################33
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))
def approxy_by_quad(cnt): 
    point_nums = cnt.shape[0] 
    approx = cnt.copy() 
    r = 0.1
    iteration = 0
    while point_nums != 4: 
        epsilon = cv2.arcLength(cnt, True) 
        approx = cv2.approxPolyDP(cnt, r*epsilon, True) 
        point_nums = approx.shape[0] 
        if point_nums > 4: 
            r *= 2
        if point_nums < 4: 
            r /= 2 
        iteration += 1
        if iteration > 1000:
            break
    return approx
 
def remove_border2(I, return_borders=False):
    Mask_w = (I > 199).astype(np.uint8)
    Mask_b = (I == 0).astype(np.uint8)
    Mask_wb = cv2.bitwise_or(Mask_w, Mask_b)
    u = Mask_wb[0,:]
    b = Mask_wb[-1,:]
    l = Mask_wb[:,0]
    r = Mask_wb[:,-1]
    Mask_wb = cv2.morphologyEx(Mask_wb, cv2.MORPH_ERODE, np.ones((5,5)), 1) 
    Mask_wb[0,:] = u
    Mask_wb[-1,:] = b
    Mask_wb[:,0] = l
    Mask_wb[:,-1] = r
    Mask = 1 - Mask_wb
    cnt, _ = cv2.findContours(Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = I.shape
    A = [cv2.contourArea(cnt[i]) for i in range(len(cnt))]
    idx = A.index(max(A))
    #x, y, w, h = cv2.boundingRect(cnt[idx])
    c = approxy_by_quad(cnt[idx])
    
    if c.shape[0] == 4:
    
        x = c[c[:,0,0].argsort()[1],0,0]
        y = c[c[:,0,1].argsort()[1], 0, 1]
        x2 = c[c[:,0,0].argsort()[-2], 0,0] 
        y2 = c[c[:,0,1].argsort()[-2], 0, 1]
        x += 2 if x !=0 else 0 
        y += 2 if y !=0 else 0
        x2 -= 2 if x2 != W-1 else 0
        y2 -= 2 if y2 != H-1 else 0 
    else:
        x = 0
        y = 0
        x2 = W
        y2 = H
    if np.prod(I[y:y2, x:x2].shape) > np.prod(I.shape) / 10:
        I = I[y:y2, x:x2]
        
    else:
        x = 0
        y = 0
        x2 = W
        y2 = H


    if return_borders is True:
        return I, [x, y, x2, y2]
    else:
        return I
###########################
def edgedetector(T, C=10):
    ff = np.zeros((2,1,1,1))
    ff[0,:] = -1
    ff[1,:] = 1
    ff = np.concatenate([ff]*C, 2)
    ff = tf.constant(ff, dtype=tf.float32)
    O1 = tf.nn.relu(-tf.nn.depthwise_conv2d(T, ff, [1,1,1,1], 'SAME'))[:,:,:,0:-1]

    ff = np.zeros((2,1,1,1))
    ff[0,:] = 1
    ff[1,:] = -1
    ff = np.concatenate([ff]*C, 2)
    ff = tf.constant(ff, dtype=tf.float32)
    O2 = tf.nn.relu(-tf.nn.depthwise_conv2d(T, ff, [1,1,1,1], 'SAME'))[:,:,:,1::]

    O = (O1 + O2) / 2

    return O


def get_boundary(e):
    e = e.argmax(1)[0]
    for i in range(e.shape[-1]):
        e[:,i] = tfa.median_filter2d(np.expand_dims(np.expand_dims(np.expand_dims(e[:,i],0),-1),-1),(11, 1), padding='REFLECT').numpy()[0,:,0,0]
        #e[:,i] = smooth(e[:,i])
    return e
##########################################################################
def ComplexLoss10(y_true, y_pred):
    return edge_loss(y_true, y_pred, C=10) + 2*focal_tversky_loss(y_true, y_pred)

def ComplexLoss3(y_true, y_pred):
    return edge_loss(y_true, y_pred, C=3) + 2*focal_tversky_loss(y_true, y_pred) 

#####################################################################33
class DUNET(object):
    def __init__(self, ds=2):
        self.bg_model, self.model2 = get_model(ds)
        #self.model_lower = get_model(input_img, 3,n_filters=32, dropout=0.05, batchnorm=True)
        #self.model_lower.load_weights('model-oct-best1.h5')
        self.layer_name = ['Inner Limiting Membrane', 'RNFL - GCL interface', 'GCL - IPL interface', 'IPL - INL interface', 'INL - OPL interface', 'OPL - ONL interface', 'External Limiting Membrane', 'Inner boundary of EZ', 'Inner boundary of RPE/IZ complex', "Bruch's Membrane", 'Choroid - sclera interface']
        inputs = tf.keras.layers.Input((None, None, 1))
        self.model = tf.keras.models.Model(inputs, [self.bg_model(inputs), self.model2(inputs)])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8), loss=[ComplexLoss3, ComplexLoss10], loss_weights=[1,1],  metrics=['acc'])
        self.model.load_weights('DBM_partial_full_d4.h5')
    def predictPNG(self, name):
        if type(name) == str:
            I = cv2.imread(name, 0)
        elif type(name) == type(np.array([1,2])):
            I = name.copy()
        jmp, o = remove_border2(I, True)
        x1,y1, x2, y2 = o
        X = np.expand_dims(jmp, [0,-1])/255
        
        t, p = self.model.predict(X)
        T = np.zeros((1, I.shape[0], I.shape[1], 3), np.float32)
        P = np.zeros((1, I.shape[0], I.shape[1], 10), np.float32)
        valid_x = np.arange(x1, x2)
        P[0, y1:y2, x1:x2] = p
        T[0, y1:y2, x1:x2] = t
        
        ep = edgedetector(p, C=10)
        if type(ep) != np.ndarray:
            ep = ep.numpy()
        e = get_boundary(ep[:,:,:,:])+y1
        
        J = {}
        kk = 0
        for i in range(10):
            if i != 2:
                K = {'X':valid_x.tolist(), 'Y':e[:,kk].tolist()}
                J[self.layer_name[i]] = K
                kk+=1
            else:
                K = {'X':[], 'Y':[]}
                J[self.layer_name[i]] = K

        return J, P, T
    
    def predictVOL(self, name):
        if type(name) != str:
            print('Error, Enter cannot find VOL file')
            return -1
        
        O = OctRead(name)
        hdr = O.get_oct_hdr()
        B = O.get_b_scans(hdr)
        N = B.shape[-1]
        P = np.zeros((N, B.shape[0], B.shape[1], 10))
        #######
        J = {}
        for i in range(N):
            I = B[...,i].copy()
            js, p, _ = self.predictPNG(I)
            J[i] = js
            P[i] = p.copy()
        return J, np.moveaxis(B, -1, 0), P
    
    def predict(self, name):
        if type(name) == str:
            if name[-3:] == 'png':
                J, P, T = self.predictPNG(name)
                return J, P, T
            elif name[-3:] == 'vol':
                J, B, P = self.predictVOL(name)
                return J, B, P
            else:
                print('I cannot understand your input! for string input, it must ends with png or vol')
                return -1
        elif type(name) == np.ndarray:
            J, P, T = self.predictPNG(name)
            return J, P, T
        else:
            print('Invalid input, it must np.ndarray for single image or string')
            return -1
            
    
        
