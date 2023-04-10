import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import backend as K

NUM_TARGET_FIDS=13
NUM_TARGET_CONCS=12
NUM_TRAIN_SAMPLES=100,000
BATCH_SIZE=16
NUM_EPOCHS=15
LEARNING_RATE=5e-4

# Default distribution strategy for single GPU is:
#    strategy = tf.distribute.get_strategy()
# if TPU, use the following code to creat a distributed strategy on a tpu:
#    tf.config.experimental_connect_to_cluster(tpu)
#    tf.tpu.experimental.initialize_tpu_system(tpu)
#    strategy = tf.distribute.experimental.TPUStrategy(tpu)
#
# The input for training is a tuple ({'FID input':total_signal}, target). The target here is a dictionary {'concentrations':concentrations, 'target_individual_signals':
# individual_signals, 'target_tatal_signal': total_signal, 'phase':phase, 'frequency':frequency}. Different from that in FID input, the target total signal is free of
# noise and extraneous peaks in order to compute total FID loss (which can be disabled by setting the loss_weight = 0). All item values in the dictionaries are 
# tf.float32 tensors and have the unbatched formats:(32, 2048, 2), (NUM_TARGET_CONCS), (32, 2048, NUM_TARGET_FIDS*2), (32, 2048, 2), (2), (2), respectively. The phase
# means (cos(angle), sin(angle)). The two values in the frequency are the frequency offsets of water and metabolites, respectively. Performing inference only needs the
# input {'FID input': total_signal}. The key names in the target dictionary need to match the output names specified in the model. Use tf Dataset to prepare data and
# tf Dataloader to batch and load data. The final FID input has the format (BATCH_SIZE, 32, 2048, 2).


print("REPLICAS: ", strategy.num_replicas_in_sync)
REPLICA_BATCH_SIZE=BATCH_SIZE//strategy.num_replicas_in_sync

def wavenet(x, filters, dilations, kernel_size=5):
    dilation_rates = [2**i for i in range(dilations)]
    x = L.Conv1D(filters = filters, 
                            kernel_size = 1,
                            padding = 'same')(x)                                                             
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = L.Conv1D(filters = filters,
                      kernel_size = kernel_size,
                      padding = 'same', 
                      activation = 'tanh', 
                      dilation_rate = dilation_rate)(x)
        sigm_out = L.Conv1D(filters = filters,
                      kernel_size = kernel_size,
                      padding = 'same',
                      activation = 'sigmoid', 
                      dilation_rate = dilation_rate)(x)
        x = L.Multiply()([tanh_out, sigm_out])

        x = L.Conv1D(filters = filters,   kernel_size = 1,padding = 'same')(x)
        res_x = L.Add()([res_x, x])
    return res_x

def deepJPRESS(dims, echoes=32, points=2048, dilation_depth=8,
               num_concentrations=NUM_TARGET_CONCS, num_fids=NUM_TARGET_FIDS):
  
          input = L.Input(shape=(echoes, points, 2), name='FID input')
          x = tf.reshape(input, (-1,points,2))

          #encoder block1 
          x = wavenet(x, dims, dilation_depth)
          x = L.LayerNormalization(axis=-1)(x)
          wavenet_out = L.Activation('relu')(x)
          to_decoder1 = tf.stop_gradient(wavenet_out)
        
          x = L.GlobalAveragePooling1D()(wavenet_out)
          x = tf.reshape(x,(-1,echoes, dims))
          x = L.Bidirectional(L.GRU(dims,return_sequences=True))(x)
          x = L.Conv1D(dims, kernel_size=1, padding='same', use_bias=False)(x)
          x = L.LayerNormalization(axis=-1)(x)
          x = L.Activation('relu')(x)    
          x = tf.reshape(x, (-1, dims))
          x = L.RepeatVector(2048)(x) 
          x = tf.concat([wavenet_out, x],-1)

          #encoder block2 
          x = wavenet(x, dims, 8)
          x = L.LayerNormalization(axis=-1)(x)
          wavenet_out = L.Activation('relu')(x)
          to_decoder2 = tf.stop_gradient(wavenet_out)

          x = L.GlobalAveragePooling1D()(wavenet_out)
          x = tf.reshape(x,(-1,echoes, dims))
          x = L.Bidirectional(L.GRU(dims, return_sequences=True))(x)
          x = L.Conv1D(dims, kernel_size=1, padding='same', use_bias=False)(x)
          x = L.LayerNormalization(axis=-1)(x)
          x = L.Activation('relu')(x)
          x = tf.reshape(x, (-1, dims))
          x = L.RepeatVector(points)(x)
          x = tf.concat([wavenet_out, x],-1)

          #encoder block3 
          x = wavenet(x, dims, dilation_depth)
          x = L.LayerNormalization(axis=-1)(x)
          wavenet_out = L.Activation('relu')(x)

          x = L.GlobalAveragePooling1D()(wavenet_out)
          x = tf.reshape(x,(-1,echoes, dims))
          x = L.Bidirectional(L.GRU(dims,return_sequences=True))(x)
          x = L.Bidirectional(L.GRU(dims,return_sequences=True))(x)
          x = L.Conv1D(dims, kernel_size=1, padding='same', use_bias=False)(x)
          x = L.LayerNormalization(axis=-1)(x)
          x = L.Activation('relu')(x)

          #outputs of concentration, phase, and frequency
          feature = L.GlobalAveragePooling1D()(x)     
          concentrations = L.Dense(NUM_TARGET_CONCS, name='concentrations')(feature)  
          phase = L.Dense(2, name='phase')(feature)
          frequency = L.Dense(2, name='frequency')(feature)
          
          #decoder 
          x = tf.concat([wavenet_out, to_decoder2],-1)
          x = wavenet(x, dims, dilation_depth)
          x = L.LayerNormalization(axis=-1)(x)
          x = L.Activation('relu')(x)

          x = tf.concat([x, to_decoder1],-1)
          x = wavenet(x, dims, dilation_depth)
          x = L.LayerNormalization(axis=-1)(x)
          x = L.Activation('relu')(x)
          
          #outputs of individual FIDs and total FID
          x = tf.reshape(x,(-1,echoes, points, dims))
          target_total_signal = L.Dense(2, name='target_total_signal')(x)
          target_individual_signals = L.Dense(NUM_TARGET_FIDS*2, name='target_individual_signals')(x)
          x = tf.reduce_mean(x, axis=1)
  
          model = tf.keras.Model(inputs=input, outputs= [target_total_signal, target_individual_signals, frequency, phase, concentrations])

          return model

class OneCycleSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.lr_start = 1e-8
    self.lr_min = 1e-8
    self.warmup_epochs = 3
    self.hold_epochs = 7
    self.total_epochs = NUM_EPOCHS

  @tf.function  
  def __call__(self, step):
    lr_start = self.lr_start
    lr_min = self.lr_min
    lr_max = self.learning_rate
    warmup_epochs = self.warmup_epochs
    hold_epochs = self.hold_epochs
    total_epochs = self.total_epochs

    warmup_steps = warmup_epochs*(NUM_TRAIN_SAMPLES/BATCH_SIZE)
    hold_steps = hold_epochs*(NUM_TRAIN_SAMPLES/BATCH_SIZE)
    total_steps = total_epochs*(NUM_TRAIN_SAMPLES/BATCH_SIZE)

    step = tf.cast(step, tf.float32)
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_steps:
        lr = lr_max
    else:      
        progress = (step - warmup_steps - hold_steps) / (total_steps - warmup_steps - hold_steps)
        lr = lr_max * tf.math.exp(-6.0*progress)
        
    lr = tf.math.maximum(lr_min, lr)
    return lr

        
def make_model(dims=128):

    K.clear_session()
    with strategy.scope():
        def total_signal_loss(y_true, y_pred):
          y_true = tf.reshape(y_true, (y_true.shape[0], -1))
          y_pred = tf.reshape(y_pred, (y_true.shape[0], -1))
          sig_loss_replica = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
          return tf.reduce_sum(sig_loss_replica(y_true, y_pred))/REPLICA_BATCH_SIZE

        def individual_signal_loss(y_true, y_pred):
          weight = tf.constant(2*[0.1]  +  2*(NUM_TARGET_FIDS-1)*[1.0])
          weight = tf.reshape(weight, (1,1,1,NUM_TARGET_FIDS*2))
          y_true  = y_true*weight
          y_pred = y_pred*weight
          y_true = tf.reshape(y_true, (y_true.shape[0], -1))
          y_pred = tf.reshape(y_pred, (y_true.shape[0], -1))
          sig_loss_replica = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
          return tf.reduce_sum(sig_loss_replica(y_true, y_pred))/REPLICA_BATCH_SIZE

        def frq_loss(y_true, y_pred):
          frq_loss_replica = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
          return tf.reduce_sum(frq_loss_replica(y_true, y_pred))/REPLICA_BATCH_SIZE

        def phase_loss(y_true, y_pred):
          phase_loss_replica = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
          return tf.reduce_sum(phase_loss_replica(y_true, y_pred))/REPLICA_BATCH_SIZE

        def conc_loss(y_true, y_pred): 
          weight = tf.constant([0.05] + (NUM_TARGET_CONCS-1)*[1.0])
          weight = tf.reshape(weight, (1,-1))
          y_true = weight*y_true
          y_pred = weight*y_pred
          met_loss_replica = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
          return tf.reduce_sum(met_loss_replica(y_true, y_pred))/REPLICA_BATCH_SIZE
     
        opt = tf.keras.optimizers.Adam(learning_rate= OneCycleSchedule(LEARNING_RATE))
        model =deepJPRESS(dims)   
        model.compile(optimizer=opt, loss=[total_signal_loss, individual_signal_loss, frq_loss, phase_loss, conc_loss],
                        loss_weights=[0.0, 40.0, 0.0,0.0,1.0])

        model.summary()
        return model
