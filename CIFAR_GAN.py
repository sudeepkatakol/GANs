import numpy as np
import random
import math
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


X = []
for i in range(1, 6):
    print(i)
    x1 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_'+str(i))
    x2 = x1[b'data']
    X = X + x2.tolist()
x1 = unpickle('cifar-10-python/cifar-10-batches-py/test_batch')
x2 = x1[b'data']
X = X + x2.tolist()


X = np.array(X)

X = np.reshape(X, (-1, 3, 32, 32))

X = X.transpose(0, 2, 3, 1)

X = X.astype(np.uint8)


def get_data(X, BATCH_SIZE):
    index = 0
    while True:
        b = random.choice([True, False])
        if index + BATCH_SIZE <= len(X):
            data = X[index:index+BATCH_SIZE]
            index += BATCH_SIZE
        else:
            np.random.shuffle(X)
            data = X[:BATCH_SIZE]
            index = BATCH_SIZE
        if b: 
            data = np.flip(data, axis = 2)
        yield data


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = img[:, :, :]
    return image



class Generator():
    def __init__(self, batch_size, noise_dim = 128):
        self.params = []
        self.params_by_name = {}
        self.noise_dim = noise_dim
        self.name = "Generator"
        self.batch_size = batch_size
        self.CHANNEL_DIM = 64
        self.IMAGE_DIM = 4
        self.KERNEL_SIZE = 5
        self.BETA = 0.9
        self.build()
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def build(self):
        self.Var_Linear('Linear', self.noise_dim, self.IMAGE_DIM*self.IMAGE_DIM*self.CHANNEL_DIM)
        self.Var_BatchNorm('BN.1', (self.IMAGE_DIM, self.IMAGE_DIM, self.CHANNEL_DIM))
        
        self.Var_Deconv2D('Deconv1', self.CHANNEL_DIM, self.CHANNEL_DIM//2, self.KERNEL_SIZE)
        self.Var_BatchNorm('BN.2', (2*self.IMAGE_DIM, 2*self.IMAGE_DIM, self.CHANNEL_DIM//2))
        
        self.Var_Deconv2D('Deconv2', self.CHANNEL_DIM//2, self.CHANNEL_DIM//4, self.KERNEL_SIZE)
        self.Var_BatchNorm('BN.3', (4*self.IMAGE_DIM, 4*self.IMAGE_DIM, self.CHANNEL_DIM//4))
        
        self.Var_Deconv2D('Deconv3', self.CHANNEL_DIM//4, 3, self.KERNEL_SIZE)
        
    
    def Var_Linear(self, layer_name, input_dim, output_dim):
        with tf.variable_scope(self.name+'.'+layer_name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.params_by_name[self.name+'.'+layer_name+'.W'] = W
        self.params_by_name[self.name+'.'+layer_name+'.b'] = b
        self.params.append(W)
        self.params.append(b)
        
    def Var_Deconv2D(self, layer_name, input_dim, output_dim, filter_size):
        with tf.variable_scope(self.name+'.'+layer_name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [filter_size, filter_size, output_dim, input_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.params_by_name[self.name+'.'+layer_name+'.W'] = W
        self.params_by_name[self.name+'.'+layer_name+'.b'] = b
        self.params.append(W)
        self.params.append(b)
        
    def Var_BatchNorm(self, layer_name, input_shape):
        with tf.variable_scope(self.name+'.'+layer_name, reuse=tf.AUTO_REUSE):
            scale = tf.get_variable('scale', input_shape, initializer=tf.constant_initializer(1), dtype=tf.float32)
            offset = tf.get_variable('offset', input_shape, initializer=tf.constant_initializer(0), dtype=tf.float32)
            moving_variance = tf.get_variable('moving_variance', input_shape, initializer=tf.constant_initializer(1), dtype=tf.float32)
            moving_mean = tf.get_variable('moving_mean', input_shape, initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.params_by_name[self.name+'.'+layer_name+'.scale'] = scale
        self.params_by_name[self.name+'.'+layer_name+'.offset'] = offset
        self.params_by_name[self.name+'.'+layer_name+'.moving_mean'] = moving_mean
        self.params_by_name[self.name+'.'+layer_name+'.moving_variance'] = moving_variance
        self.params.append(scale)
        self.params.append(offset)
    
    def Linear(self, layer_name, inputs):
        W = self.params_by_name[self.name+'.'+layer_name+'.W']
        b = self.params_by_name[self.name+'.'+layer_name+'.b']
        return tf.matmul(inputs, W) + b
        
    def Deconv2d(self, layer_name, output_shape, inputs):
        W = self.params_by_name[self.name+'.'+layer_name+'.W']
        b = self.params_by_name[self.name+'.'+layer_name+'.b']
        result = tf.nn.conv2d_transpose(value=inputs, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME') + b
        return result
    
    def BatchNorm(self, layer_name, inputs, is_training):
        scale = self.params_by_name[self.name+'.'+layer_name+'.scale']
        offset = self.params_by_name[self.name+'.'+layer_name+'.offset']
        moving_mean = self.params_by_name[self.name+'.'+layer_name+'.moving_mean']
        moving_variance = self.params_by_name[self.name+'.'+layer_name+'.moving_variance']
        if is_training == True:
            mean, variance = tf.nn.moments(inputs, [0], keep_dims=False)
            moving_mean = self.BETA * moving_mean + (1 - self.BETA) * mean
            moving_variance = self.BETA * moving_variance + (1 - self.BETA) * variance
            return tf.nn.batch_normalization(inputs, mean, variance, offset, scale, 1e-8)
        else:
            return tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset, scale, 1e-8)
    
    def output(self, noise = None, is_training = True):
        if noise is None:
            noise = tf.random_normal((self.batch_size, self.noise_dim))
        
        X = self.Linear('Linear', noise)
        X = tf.reshape(X, (-1, self.IMAGE_DIM, self.IMAGE_DIM, self.CHANNEL_DIM))
        X = self.BatchNorm('BN.1', X, is_training)
        X = tf.nn.relu(X)
        
        output_shape = (self.batch_size, 2*self.IMAGE_DIM, 2*self.IMAGE_DIM, self.CHANNEL_DIM//2)
        X = self.Deconv2d('Deconv1', output_shape, X)
        X = self.BatchNorm('BN.2', X, is_training)
        X = tf.nn.relu(X)
        
        output_shape = (output_shape[0], output_shape[1]*2, output_shape[2]*2, output_shape[3]//2)
        X = self.Deconv2d('Deconv2', output_shape, X)
        X = self.BatchNorm('BN.3', X, is_training)
        X = tf.nn.relu(X)
        
        output_shape = (output_shape[0], output_shape[1]*2, output_shape[2]*2, 3)
        X = self.Deconv2d('Deconv3', output_shape, X)
        X = tf.tanh(X)
        
        return X

class Discriminator():
    def __init__(self):
        self.params = []
        self.params_by_name = {}
        self.name = "Discriminator"
        self.CHANNEL_DIM = 16
        self.IMAGE_DIM = 4
        self.KERNEL_SIZE = 5
        self.build()
    
    def build(self):
        self.Var_Conv2D('Conv1', 3, self.CHANNEL_DIM, self.KERNEL_SIZE)
        self.Var_Conv2D('Conv2', self.CHANNEL_DIM, 2*self.CHANNEL_DIM, self.KERNEL_SIZE)
        self.Var_Conv2D('Conv3', 2*self.CHANNEL_DIM, 4*self.CHANNEL_DIM, self.KERNEL_SIZE)
        self.Var_Linear('Linear', self.IMAGE_DIM*self.IMAGE_DIM*4*self.CHANNEL_DIM, 1)
    
    def LeakyReLU(self, x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    
    def Var_Linear(self, layer_name, input_dim, output_dim):
        with tf.variable_scope(self.name+'.'+layer_name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.params_by_name[self.name+'.'+layer_name+'.W'] = W
        self.params_by_name[self.name+'.'+layer_name+'.b'] = b
        self.params.append(W)
        self.params.append(b)
        
    def Var_Conv2D(self, layer_name, input_dim, output_dim, filter_size):
        with tf.variable_scope(self.name+'.'+layer_name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [filter_size, filter_size, input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.params_by_name[self.name+'.'+layer_name+'.W'] = W
        self.params_by_name[self.name+'.'+layer_name+'.b'] = b
        self.params.append(W)
        self.params.append(b)
        
    def Linear(self, layer_name, inputs):
        W = self.params_by_name[self.name+'.'+layer_name+'.W']
        b = self.params_by_name[self.name+'.'+layer_name+'.b']
        return tf.matmul(inputs, W) + b
        
    def Conv2d(self, layer_name, inputs):
        W = self.params_by_name[self.name+'.'+layer_name+'.W']
        b = self.params_by_name[self.name+'.'+layer_name+'.b']
        result = tf.nn.conv2d(inputs, W, [1, 2, 2, 1],padding='SAME') + b
        return result
    
    def output(self, X):
        X = self.Conv2d('Conv1', X)
        X = self.LeakyReLU(X)
        
        X = self.Conv2d('Conv2', X)
        X = self.LeakyReLU(X)
        
        X = self.Conv2d('Conv3', X)
        X = self.LeakyReLU(X)
        
        X = tf.reshape(X, [-1, self.IMAGE_DIM*self.IMAGE_DIM*4*self.CHANNEL_DIM])
        X = self.Linear('Linear', X)
        return tf.reshape(X, [-1])


LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64

real_data_uint = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3])
real_data =  2*((tf.cast(real_data_uint, tf.float32)/255.)-.5)

is_training = tf.placeholder(tf.bool, shape=())

generator = Generator(BATCH_SIZE)
fake_data = generator.output(is_training=is_training)

discriminator = Discriminator()
disc_real = discriminator.output(real_data)
disc_fake = discriminator.output(fake_data)

gen_params = generator.params
disc_params = discriminator.params

gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

alpha = tf.random_uniform(shape=[BATCH_SIZE, 32, 32, 3], minval=0.,maxval=1.)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(discriminator.output(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


PREV_ITERS = 5000


disc_losses = []
gen_losses = []
generations = []


saver = tf.train.Saver()


session = tf.Session()
# session.run(tf.global_variables_initializer())
saver.restore(session, 'models/cifar-4999')

fetch_data = get_data(X, BATCH_SIZE)

ITERS = 55000

for iteration in range(ITERS):
    for i in range(CRITIC_ITERS):
        _data = next(fetch_data)
        _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_uint: _data,                                                                           is_training: True})
    disc_losses.append(_disc_cost)
    _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={is_training: True})
    if iteration % 100 == 99:
        print("Iteration:  " + str(iteration) + " Discriminator loss " + str(_disc_cost))
        print("Iteration:  " + str(iteration) + " Generator loss " + str(_gen_cost))
    gen_losses.append(_gen_cost)
    if iteration % 500 == 499:
        generations.append(session.run(fake_data))
    if iteration % 5000 == 4999:
        saver.save(session, './models/cifar', global_step=PREV_ITERS+iteration)
        np.save('./models/cifar_disc_losses_'+str(PREV_ITERS+iteration), disc_losses)
        np.save('./models/cifar_gen_losses_'+str(PREV_ITERS +iteration), gen_losses)
        np.save('./models/cifar_generations_'+str(PREV_ITERS +iteration), generations)
PREV_ITERS += ITERS


