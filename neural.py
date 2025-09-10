from codes import ToricCode
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Nadam
from keras.losses import binary_crossentropy
from keras.layers import BatchNormalization
import tensorflow as tf

F = lambda _: tf.cast(_, 'float32') # TODO XXX there must be a better way to calculate mean than this cast-first approach


class CodeCosts:
    def __init__(self, L, code, Z, X, normcentererr_p=None):
        if normcentererr_p:
            raise NotImplementedError('Throughout the entire codebase, the normalization and centering of the error, might be wrong... Or to be more precise, it might just be plain stupid, given that we are using binary crossentropy as loss.')
        self.L = L
        code = code(L)
        H = code.H(Z,X)
        E = code.E(Z,X)
        self.H = tf.Variable(initial_value=H, trainable=False, dtype=tf.float32) # TODO should be sparse
        self.E = tf.Variable(initial_value=E, trainable=False, dtype=tf.float32) # TODO should be sparse
        self.p = normcentererr_p
    def exact_reversal(self, y_true, y_pred):
        "Fraction exactly predicted qubit flips."
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return tf.reduce_mean(F(tf.reduce_all(tf.equal(y_true, tf.round(y_pred)), axis=-1)))
    def non_triv_stab_expanded(self, y_true, y_pred):
        "Whether the stabilizer after correction is not trivial."
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        # Cast to same dtype to avoid type mismatch
        y_pred_rounded = tf.cast(tf.round(y_pred), tf.float32)
        y_true_cast = tf.cast(y_true, tf.float32)
        correction = tf.cast((y_pred_rounded + y_true_cast) % 2, tf.float32)
        return tf.reduce_any(tf.cast(tf.matmul(self.H, tf.transpose(correction)) % 2, tf.bool), axis=0)
    def logic_error_expanded(self, y_true, y_pred):
        "Whether there is a logical error after correction."
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        # Cast to same dtype to avoid type mismatch
        y_pred_rounded = tf.cast(tf.round(y_pred), tf.float32)
        y_true_cast = tf.cast(y_true, tf.float32)
        correction = tf.cast((y_pred_rounded + y_true_cast) % 2, tf.float32)
        return tf.reduce_any(tf.cast(tf.matmul(self.E, tf.transpose(correction)) % 2, tf.bool), axis=0)
    def triv_stab(self, y_true, y_pred):
        "Fraction trivial stabilizer after corrections."
        return 1-tf.reduce_mean(F(self.non_triv_stab_expanded(y_true, y_pred)))
    def no_error(self, y_true, y_pred):
        "Fraction no logical errors after corrections."
        return 1-tf.reduce_mean(F(self.logic_error_expanded(y_true, y_pred)))
    def triv_no_error(self, y_true, y_pred):
        "Fraction with trivial stabilizer and no error."
        # TODO XXX Those casts (the F function) should not be there! This should be logical operations
        triv_stab = 1 - F(self.non_triv_stab_expanded(y_true, y_pred))
        no_err    = 1 - F(self.logic_error_expanded(y_true, y_pred))
        return tf.reduce_mean(no_err*triv_stab)
    def e_binary_crossentropy(self, y_true, y_pred):
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred), axis=-1)
    def s_binary_crossentropy(self, y_true, y_pred):
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        # Cast to avoid type mismatch
        y_true_cast = tf.cast(y_true, tf.float32)
        s_true = tf.cast(tf.matmul(y_true_cast, tf.transpose(self.H)) % 2, tf.float32)
        twopminusone = 2*y_pred-1
        s_pred = ( 1 - tf.math.real(tf.exp(tf.matmul(tf.math.log(tf.cast(twopminusone, tf.complex64)), tf.cast(tf.transpose(self.H), tf.complex64)))) ) / 2
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(s_true, s_pred), axis=-1)
    def se_binary_crossentropy(self, y_true, y_pred):
        return 2./3.*self.e_binary_crossentropy(y_true, y_pred) + 1./3.*self.s_binary_crossentropy(y_true, y_pred)


def create_model(L, hidden_sizes=[4], hidden_act='tanh', act='sigmoid', loss='binary_crossentropy',
                 Z=True, X=False, learning_rate=0.002,
                 normcentererr_p=None, batchnorm=0):
    in_dim = L**2 * (X+Z)
    out_dim = 2*L**2 * (X+Z)
    model = Sequential()
    model.add(Dense(int(hidden_sizes[0]*out_dim), input_dim=in_dim, kernel_initializer='glorot_uniform'))
    if batchnorm:
        model.add(BatchNormalization(momentum=batchnorm))
    model.add(Activation(hidden_act))
    for s in hidden_sizes[1:]:
        model.add(Dense(int(s*out_dim), kernel_initializer='glorot_uniform'))
        if batchnorm:
            model.add(BatchNormalization(momentum=batchnorm))
        model.add(Activation(hidden_act))
    model.add(Dense(out_dim, kernel_initializer='glorot_uniform'))
    if batchnorm:
        model.add(BatchNormalization(momentum=batchnorm))
    model.add(Activation(act))
    c = CodeCosts(L, ToricCode, Z, X, normcentererr_p)
    losses = {'e_binary_crossentropy':c.e_binary_crossentropy,
              's_binary_crossentropy':c.s_binary_crossentropy,
              'se_binary_crossentropy':c.se_binary_crossentropy}
    model.compile(loss=losses.get(loss,loss),
                  optimizer=Nadam(learning_rate=learning_rate),
                  metrics=[c.triv_no_error, c.e_binary_crossentropy, c.s_binary_crossentropy]
                 )
    return model

def makeflips(q, out_dimZ, out_dimX):
    flips = np.zeros((out_dimZ+out_dimX,), dtype=np.dtype('b'))
    rand = np.random.rand(out_dimZ or out_dimX) # if neither is zero they have to necessarily be the same (equal to the number of physical qubits)
    both_flips  = (2*q<=rand) & (rand<3*q)
    if out_dimZ: # non-trivial Z stabilizer is caused by flips in the X basis
        x_flips =                rand<  q
        flips[:out_dimZ] ^= x_flips
        flips[:out_dimZ] ^= both_flips
    if out_dimX: # non-trivial X stabilizer is caused by flips in the Z basis
        z_flips =   (q<=rand) & (rand<2*q)
        flips[out_dimZ:out_dimZ+out_dimX] ^= z_flips
        flips[out_dimZ:out_dimZ+out_dimX] ^= both_flips
    return flips

def nonzeroflips(q, out_dimZ, out_dimX):
    flips = makeflips(q, out_dimZ, out_dimX)
    while not np.any(flips):
        flips = makeflips(q, out_dimZ, out_dimX)
    return flips

def data_generator(H, out_dimZ, out_dimX, in_dim, p, batch_size=512, size=None,
                   normcenterstab=False, normcentererr=False):
    c = 0
    q = (1-p)/3
    while True:
        flips = np.empty((batch_size, out_dimZ+out_dimX), dtype=int) # TODO dtype? byte?
        for i in range(batch_size):
            flips[i,:] = nonzeroflips(q, out_dimZ, out_dimX)
        stabs = np.dot(flips,H.T)%2
        if normcenterstab:
            stabs = do_normcenterstab(stabs, p)
        if normcentererr:
            flips = do_normcentererr(flips, p)
        yield (stabs, flips)
        c += 1
        if size and c >= size:
            return

def do_normcenterstab(stabs, p):
    avg = (1-p)*2/3
    avg_stab = 4*avg*(1-avg)**3 + 4*avg**3*(1-avg)
    var_stab = avg_stab-avg_stab**2
    return (stabs - avg_stab)/var_stab**0.5

def undo_normcenterstab(stabs, p):
    avg = (1-p)*2/3
    avg_stab = 4*avg*(1-avg)**3 + 4*avg**3*(1-avg)
    var_stab = avg_stab-avg_stab**2
    return stabs*var_stab**0.5 + avg_stab

def do_normcentererr(flips, p):
    avg = (1-p)*2/3
    var = avg-avg**2
    return (flips-avg)/var**0.5

def undo_normcentererr(flips, p):
    avg = (1-p)*2/3
    var = avg-avg**2
    return flips*var**0.5 + avg

def smart_sample(H, stab, pred, sample, giveup):
    '''Sample `pred` until `H@sample==stab`.

    `sample` is modified in place. `giveup` attempts are done at most.
    Returns the number of attempts.'''
    npany = np.any
    nprandomuniform = np.random.uniform
    npsum = np.sum
    npdot = np.dot
    attempts = 1
    mismatch = stab!=npdot(H,sample)%2
    while npany(mismatch) and attempts < giveup:
        propagated = npany(H[mismatch,:], axis=0)
        sample[propagated] = pred[propagated]>nprandomuniform(size=npsum(propagated))
        mismatch = stab!=npdot(H,sample)%2
        attempts += 1
    return attempts
