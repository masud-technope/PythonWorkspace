import tensorflow as tf
import numpy as np
import re
import tensorflow as tf
import time
import unicodedata

# a = tf.constant([[1, 2],[3, 4]])
# b = tf.matmul(a, a)
# print(b)

sentences = [
    ("Do you want a cup of coffee?", "¿Quieres una taza de café?"),
    ("I've had coffee already.", "Ya tomé café."),
    ("Can I get you a coffee?", "¿Quieres que te traiga un café?"),
    ("Please give me some coffee.", "Dame algo de café por favor."),
    ("Would you like me to make coffee?", "¿Quieres que prepare café?"),
    ("Two coffees, please.", "Dos cafés, por favor."),
    ("How about a cup of coffee?", "¿Qué tal una taza de café?"),
    ("I drank two cups of coffee.", "Me tomé dos tazas de café."),
    ("Would you like to have a cup of coffee?", "¿Te gustaría tomar una taza de café?"),
    ("There'll be coffee and cake at five.", "A las cinco habrá café y un pastel."),
    ("Another coffee, please.", "Otro café, por favor."),
    ("I made coffee.", "Hice café."),
    ("I would like to have a cup of coffee.", "Quiero beber una taza de café."),
    ("Do you want me to make coffee?", "¿Quieres que haga café?"),
    ("It is hard to wake up without a strong cup of coffee.", "Es difícil despertarse sin una taza de café fuerte."),
    ("All I drank was coffee.", "Todo lo que bebí fue café."),
    ("I've drunk way too much coffee today.", "He bebido demasiado café hoy."),
    ("Which do you prefer, tea or coffee?", "¿Qué prefieres, té o café?"),
    ("There are many kinds of coffee.", "Hay muchas variedades de café."),
    ("I will make some coffee.", "Prepararé algo de café.")
]


def preprocess(s):
    # for details, see https://www.tensorflow.org/alpha/tutorials/sequences/nmt_with_attention
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    s = s.strip()
    s = '<start> ' + s + ' <end>'
    return s


print("Original:", sentences[0])
sentences = [(preprocess(source), preprocess(target)) for (source, target) in sentences]
print("Preprocessed:", sentences[0])

source_sentences, target_sentences = list(zip(*sentences))

source_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
source_tokenizer.fit_on_texts(source_sentences)
source_data = source_tokenizer.texts_to_sequences(source_sentences)
print("Sequence:", source_data[0])
source_data = tf.keras.preprocessing.sequence.pad_sequences(source_data, padding='post')
print("Padded:", source_data[0])

target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_sentences)
target_data = target_tokenizer.texts_to_sequences(target_sentences)
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, padding='post')

target_labels = np.zeros(target_data.shape)
target_labels[:, 0:target_data.shape[1] - 1] = target_data[:, 1:]
print("Target sequence", target_data[0])
print("Target label", target_labels[0])


source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1


def decode(encoded, tokenizer):
    for number in encoded:
        if number != 0:
            print("%d -> %s" % (number, tokenizer.index_word[number]))


decode(source_data[0], source_tokenizer)

batch_size = 5
dataset = tf.data.Dataset.from_tensor_slices((source_data, target_data, target_labels)).batch(batch_size)

example_batch = next(iter(dataset))
source, target, taget_labels = example_batch
print("Shapes:", source.shape, target.shape, taget_labels.shape)
embedding_size = 32
rnn_size = 64


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(source_vocab_size,
                                                   embedding_size)
        self.gru = tf.keras.layers.GRU(rnn_size,
                                       return_sequences=True,
                                       return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def init_state(self, batch_size):
        return tf.zeros((batch_size, rnn_size))




