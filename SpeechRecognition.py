from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import concatenate
import tensorflow_probability as tfp
from tensorflow.nn import ctc_loss, ctc_greedy_decoder, ctc_beam_search_decoder
from math import ceil

class SPS(Model):
    def __init__(self, inputs, outputs, name = 'SPS'):
        super(SPS, self).__init__(inputs = inputs, outputs = outputs, name = name)
        self.loss_history = []
        self.ler_history = []
        self.val_loss_history = []
        self.val_ler_history = []
        self.trained_epochs = 0

    def predict(self, input, training = False): 
        if len(input.shape) == 2:
            input = np.expand_dims(input, 0) # if shape is [timesteps, frames] it becomes [1, timesteps, frames]

        predictions = self(input)
        logits = tf.transpose(predictions, [1, 0, 2]) # Converting the output to logits to feed it to the ctc decoder, turning the shape from [batch_size, timesteps, frames] to [timesteps, batch_size, labels]
        logits_sequence_length = [logits.shape[0] for _ in range(logits.shape[1])] # Setting an array with the batch_size number for each logit timestep, which is constant
        decoded, neg_sum_logits = self.ctc_decoder(logits, logits_sequence_length) # Decoding the neural network output
        decoded_logits = decoded[0] # The sparse tensor outputted by the ctc decoder is the first element of the list
        output = self.sparse_tensor_to_list(tf.cast(decoded_logits, tf.int32))

        return output         

    def compile(self, optimizer = RMSprop(), greedy = True, beam_width = 100, top_paths = 1):
        self.optimizer = optimizer
        self.loss_function = ctc_loss
        self.ctc_decoder = ctc_greedy_decoder if greedy == True else ctc_beam_search_decoder
        self.metric = tf.edit_distance   
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths


    def fit(self, inputs, targets, decoder_inputs = None, batch_size = 32, num_epochs = 1, val_inputs = None, val_targets = None, val_batch_size = None, teacher_forcing = False, return_history = False):
        num_train_samples = inputs.shape[0]
        num_val_samples = 0
        # Converting the dataset to tensor
#        temp_inputs = np.expand_dims(inputs, 0)
#        inputs = tf.convert_to_tensor(temp_inputs)
#        inputs = tf.squeeze(inputs)
#        targets = self.sparse_tuple_from(targets)
        
        if val_inputs is not None and val_targets is not None:
            num_val_samples = val_inputs.shape[0]
#            temp_val_inputs = np.expand_dims(val_inputs, 0)
#            val_inputs = tf.convert_to_tensor(temp_val_inputs)
#            val_inputs = tf.squeeze(val_inputs)
#            val_targets = self.sparse_tuple_from(val_targets)

            if val_batch_size is None:
                val_batch_size = batch_size         

        print(str(num_epochs) + " epoch(s) running over " + str(num_train_samples) + " train sample(s) and " + str(num_val_samples) + " validation sample(s)\n\n")
        
        for epoch in range(num_epochs):
            print("Epoch " + str(epoch + 1) + " training")
            # Making a new random batch after each epoch
            train_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(num_train_samples).batch(batch_size) 
            train_loss = 0
            train_ler = 0
            executed_inputs = 0

            # Progress bar
            progbar = Progbar(num_train_samples, stateful_metrics = ['ctc_loss', 'ler'])
            progbar.update(0, values = [('ctc_loss', np.inf), ('ler', np.inf)])

            for batch in train_dataset:
                current_batch_size = batch_size if executed_inputs + batch_size <= num_train_samples else num_train_samples - executed_inputs
                batch_inputs, batch_targets = batch
                batch_loss, batch_predictions = self.train_step(batch_inputs, batch_targets)
                batch_loss = batch_loss.numpy()
                batch_ler = self.calculate_ler(batch_targets, batch_predictions)
                executed_inputs += current_batch_size
                # Updating the progress bar
                progbar.update(executed_inputs, values = [('ctc_loss', batch_loss), ('ler', batch_ler)])
                train_loss += batch_loss
                train_ler += batch_ler

            train_loss /= ceil(num_train_samples/batch_size)
            train_ler/= ceil(num_train_samples/batch_size)
            self.loss_history.append(train_loss)
            self.ler_history.append(train_ler)
            self.trained_epochs += 1
            
            print("ctc_loss = %.3f" % train_loss + " ler = %.3f" % train_ler)

            if num_val_samples > 0: # if a validation dataset is passed
                self.evaluate(val_inputs, val_targets, batch_size = val_batch_size, training = True)
            print("\n\n")
        
        if return_history:
            return self.loss_history, self.ler_history, self.val_loss_history, self.val_ler_history

    def evaluate(self, inputs, targets, batch_size = 128, training = False):
        num_samples = len(inputs)

        # Making the validation batches        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)       

        val_loss = 0
        val_ler = 0
        executed_samples = 0

        print("Evaluating...")
        # Progress bar
        progbar = Progbar(num_samples, stateful_metrics = ['val_ctc_loss', 'val_ler'])
        progbar.update(0, values = [('val_ctc_loss', np.inf), ('val_ler', np.inf)])

        for batch_inputs, batch_targets in dataset:
            current_batch_size = batch_size if executed_samples+ batch_size <= num_samples else num_samples - executed_samples
            batch_loss, batch_ler = self.test_step(batch_inputs, batch_targets)
            executed_samples += current_batch_size
            # Updating the progress bar
            progbar.update(executed_samples, values = [('val_ctc_loss', batch_loss), ('val_ler', batch_ler)])
            val_loss += batch_loss
            val_ler += batch_ler

        val_loss /= ceil(num_samples/batch_size)
        val_ler/= ceil(num_samples/batch_size)
        
        print("val_ctc_loss = %.3f" % val_loss + " val_ler = %.3f" % val_ler)

        if training:
            self.val_loss_history.append(val_loss)
            self.val_ler_history.append(val_ler)

    def test_step(self, batch_inputs, batch_targets):
        X = self(batch_inputs, training = False)
        logits = tf.transpose(X, [1, 0, 2]) # Converting the output to logits to feed it to the ctc decoder, turning the shape from [batch_size, timesteps, labels] to [timesteps, batch_size, labels]
        logits_sequence_length = [logits.shape[0] for _ in range(logits.shape[1])] # Setting an array with the batch_size number for each logit timestep, which is constant
        batch_loss = self.loss_function(batch_targets, 
                                        logits, 
                                        [None], 
                                        logits_sequence_length, 
                                        blank_index = X.shape[2] - 1) # Calculating the CTC loss        
        average_batch_loss = tf.reduce_mean(batch_loss)
        decoded, neg_sum_logits = self.ctc_decoder(logits, logits_sequence_length) # Decoding the neural network output
        decoded_logits = decoded[0] # The sparse tensor outputted by the ctc decoder is the first element of the list
        decoded_predictions = tf.cast(decoded_logits, tf.int32)
        average_batch_ler = self.calculate_ler(batch_targets, decoded_predictions) # Calculating the LER

        return average_batch_loss.numpy(), average_batch_ler 

    #@tf.function
    def train_step(self, batch_inputs, batch_targets, batch_decoder_inputs = None):
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(self.trainable_variables)
            X = self(batch_inputs, training = True)
            logits = tf.transpose(X, [1, 0, 2]) # Converting the output to logits to feed it to the ctc decoder, turning the shape from [batch_size, timesteps, labels] to [timesteps, batch_size, labels]
            logits_sequence_length = [logits.shape[0] for _ in range(logits.shape[1])] # Setting an array with the batch_size number for each logit timestep, which is constant
            batch_loss = self.loss_function(batch_targets, 
                                            logits, 
                                            [None], 
                                            logits_sequence_length, 
                                            blank_index = X.shape[2] - 1) # Calculating the CTC loss        
        average_batch_loss = tf.reduce_mean(batch_loss)
        gradients = tape.gradient(batch_loss, self.trainable_variables) # Calculating the gradients for the neural network
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # Applying the gradients to the neural network
        decoded, neg_sum_logits = self.ctc_decoder(logits, logits_sequence_length) # Decoding the neural network output
        decoded_logits = decoded[0] # The sparse tensor outputted by the ctc decoder is the first element of the list
        decoded_predictions = tf.cast(decoded_logits, tf.int32)
        
        return average_batch_loss, decoded_predictions      

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1])

        return tf.dtypes.cast(tf.SparseTensor(indices, values, shape), dtype = dtype)

    def sparse_tensor_to_list(self, sparse_tensor):
        batch_size = sparse_tensor.dense_shape[0].numpy()
        current_batch_item = 0
        index_start = 0
        values = sparse_tensor.values.numpy()
        indices = sparse_tensor.indices.numpy()
        length = len(values)
        targets = []

        while current_batch_item < batch_size:
            current_size = 0
            target = []

            for i in range(index_start, length):
                if indices[i][0] == current_batch_item:
                    #if values[i] 
                    target.append(values[i])
                    current_size += 1
                else:
                    index_start = current_size
                    current_size = 0
                    break

            targets.append(target)
            current_batch_item += 1

        return np.array(targets)

    def calculate_ler(self, y, y_pred):
        LER = []

        list_y = self.sparse_tensor_to_list(y)
        list_y_pred = self.sparse_tensor_to_list(y_pred)

        batch_size = len(list_y)

        for y_target, y_pred_target in zip(list_y, list_y_pred):
            LER.append(self.minimumEditDistance(y_target, y_pred_target))

        return np.sum(np.asarray(LER))/batch_size
            

    def minimumEditDistance(self, y, y_pred): 
        # Adapted from https://stackoverflow.com/questions/53015099/calculating-minimum-edit-distance-for-unequal-strings-python

        #matrix = np.zeros((len(y)+1,len(y_pred)+1), dtype=np.int)
        matrix = [[(0, '') for _ in range(len(y_pred)+1)] for _ in range(len(y)+1)]
                        
        I = 0
        D = 0
        C = 0
        S = 0

        for i in range(len(y)+1): 
            for j in range(len(y_pred)+1): 

                if i == 0:  
                    matrix[i][j] = (j, 'd')  
                elif j == 0: 
                    matrix[i][j] = (i, 'i')
                else: 
                    insertion = (matrix[i-1][j][0] + 1, matrix[i-1][j][1] + 'i')
                    deletion = (matrix[i][j-1][0] + 1, matrix[i][j-1][1] + 'd')
                    substituition = (matrix[i-1][j-1][0] + 2, matrix[i-1][j-1][1] + 's') if y[i-1] != y_pred[j-1] else (np.inf, matrix[i-1][j-1][1] + 's')
                    correct = (matrix[i-1][j-1][0], matrix[i-1][j-1][1] + 'c') if y[i-1] == y_pred[j-1] else (np.inf, matrix[i-1][j-1][1] + 'c')
                    #matrix[i][j] = min(insertion, deletion, substituition, correct)


                    if insertion <= deletion and insertion <= substituition and insertion <= correct:  
                        matrix[i][j] = insertion
                    elif deletion <= insertion and deletion <= substituition and deletion <= correct:
                        matrix[i][j] = deletion
                    elif substituition <= insertion and substituition <= deletion and substituition <= correct:
                        matrix[i][j] = substituition
                    elif correct <= insertion and correct <= deletion and correct <= substituition:    
                        matrix[i][j] = correct
                    else:
                        raise ValueError("Something went wrong")

        edit_distance = matrix[len(y)][len(y_pred)][0]

        for operation in matrix[len(y)][len(y_pred)][1]:
            if operation == 'i':
                I += 1
            elif operation == 'd':
                D += 1
            elif operation == 's':
                S += 1
            elif operation == 'c':
                C += 1
            else:
                raise ValueError("Invalid operation '" + str(operation) + "'")

        LER = (S + D + I)/(S + D + C) if (S + D + C) > 0 else I

        return LER 

    def get_model(self):
        return self