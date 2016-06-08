import tensorflow as tf
import numpy as np


class LSTM(object):

    def __init__(self, is_training):

        self.batch_size = batch_size = 1
        self.num_steps = num_steps = 1
        self.num_features = num_features = 3
        self.dense_units = dense_units = 2
        self.num_layers = num_layers = 2
        self.keep_prob = keep_prob = 1
        self.max_grad_norm = max_grad_norm = 5
        self.targets = tf.placeholder(tf.float32, [dense_units])
        self._lr = tf.Variable(0.0, trainable=False)
        self._cost = 5
        size = 10

        self._input_data = inputs = tf.placeholder(tf.float32, [num_features,])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, forget_bias=0.0)

        # Wrap the cell in a dropout layer
        if is_training and keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,
                                                      output_keep_prob=keep_prob)

        # Creates a stacked model with num_layers number of lstm cells
        # Output of first layer is input of second and so on
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        # Initialize state as 2D Tensor of shape [batch_size x state_size] filled with zeros
        # State size is total number of units between all cells (i.e. size * num_layers)
        self._initial_state = stacked_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        inputs = [tf.reshape(inputs, [1, num_features])]

        # Computes dropout
        if is_training and keep_prob < 1:
            inputs = [tf.nn.dropout(x, keep_prob) for x in inputs]

        # Builds the RNN
        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        # Joins all output tensors and creates shape with width 'size'
        output = tf.reshape(tf.concat(1, outputs), shape=[-1, size])

        # Add a fully-connected layer
        self.dense_w = dense_w = tf.get_variable('dense_w', shape=[size, dense_units])
        self.dense_b = dense_b = tf.get_variable('dense_b', shape=[dense_units])

        # Feed the output from the RNN to the fully-connected layer with an activation function
        # The sum of all the activations is the predicted target
        self._predictions = predictions = tf.sigmoid(tf.matmul(output, dense_w) + dense_b)
        self._final_state = state

        self.tvars = tvars = tf.trainable_variables()
        return

    # Set new learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def update(self):

        self.qvalue = tf.placeholder(tf.float32, (2,))
        self._cost = cost = tf.reduce_mean(tf.square(tf.sub(self.predictions, self.qvalue)))
        grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(cost, self.tvars),
                                          clip_norm=self.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, self.tvars))

    @property
    def input_data(self):
        return self._input_data

    @property
    def predictions(self):
        return self._predictions

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, m, data, eval_op):

    state = m.initial_state.eval()
    prediction, state, _ = session.run(fetches=[m.predictions, m.final_state, eval_op],
                                       feed_dict={m.input_data: data,
                                                  m.initial_state: state})
    return prediction