import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
import nltk
from gensim.models.keyedvectors import KeyedVectors
import operator
import os
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import logging, os
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import *

df = pd.read_csv("/train.csv", index_col='id')
df


f,ax=plt.subplots(1,2,figsize=(18,8))
target = df['target'].value_counts().rename({0:'Unreal disaster', 1:'Real disaster'})

target.plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target distibution')
ax[0].set_ylabel('')
ax[0].legend(prop={'size': 10})

target.plot.bar(ax=ax[1], color=['tab:blue', 'tab:orange'], rot=20)
ax[1].set_title('Target distibution')
plt.show()

keywords = df['keyword'].value_counts(dropna=False)
keywords

f,ax=plt.subplots()
keywords.plot.hist(ax=ax, bins=61)
ax.set_title('Frequency of keywords occurrences')
ax.set_xlabel('Number of occurrences')
plt.show()

keywords_groupby_target = df.groupby('keyword')['target'].value_counts()
share_unreal_disasters, share_real_disasters = [], []

for keyword in df['keyword'].dropna().unique():
    keyword_value = keywords_groupby_target[keyword]
    sum_count = 0
    for target in keyword_value.index.tolist():
        sum_count += keyword_value[target]
    if 0 in keyword_value:
        share_unreal_disasters.append(keyword_value[0] / sum_count)
        share_real_disasters.append(1 - keyword_value[0] / sum_count)
    else: 
        share_unreal_disasters.append(0)
        share_real_disasters.append(1)

bins = np.linspace(0, 1, 30)
f,ax=plt.subplots(figsize=(12,8))
plt.hist(share_real_disasters, bins, label=['Real disaster'])
plt.legend(loc='upper right')
ax.set_title('Frequency Distribution of Real Disasters among Posts with a Certain Keyword')
ax.set_xlabel('Proportion of posts by keywords')
ax.set_ylabel('Frequency')
plt.show()

locations = df['location'].value_counts(dropna=False)
locations

print(list(locations.loc[lambda x : x >=5].index))

print(list(locations.loc[lambda x : x == 1].index)[:100])



def clean_location(text: str):
    result = re.sub(r'[^A-Za-z\s]', ' ', text)
    result = ' '.join([w for w in result.split() if len(w)>1])
    result = re.sub(' +', ' ', result)
    result = result.strip().lower()
    result = re.sub(r'(^|\s)(us|united states)($|\s)', ' usa ', result)
    result = re.sub(r'(^|\s)(united kingdom|england)($|\s)', ' uk ', result)
    return result.strip().lower()

df['location'] = df.apply(lambda row: pd.NA if pd.isna(row['location']) else clean_location(row['location']), axis=1)
locations = df['location'].value_counts(dropna=False)
locations

f,ax=plt.subplots(1,2, figsize=(18,8))

bins=[0, 1, 10, 20, 40, 110, 2533]
locations_count = pd.DataFrame(locations)
locations_count['bins'] = pd.cut(locations, bins=bins, labels=['NaN' if bins[i+1] == 2533 else f'{bins[i] + 1}-{bins[i+1]}' for i in range(len(bins) - 1)])
locations_count.groupby('bins').sum()['count'].plot.pie(autopct='%1.1f%%',ax=ax[0], shadow=True)
ax[0].set_title('Frequency of locations occurrences')
ax[0].set_ylabel('Number of location occurrences')

df['location'].value_counts().plot.box(ax=ax[1])
ax[1].set_title('Frequency of not NaN locations occurrences')
ax[1].set_ylabel('Number of location occurrences')
plt.show()


f,ax=plt.subplots(2,2,figsize=(12,12))

location_isna_target = df[df['location'].isna()]['target'].value_counts().rename({0:'Unreal disaster', 1:'Real disaster'})
location_isna_target.plot.bar(ax=ax[0, 0], color=['tab:blue', 'tab:orange'], rot=20)
ax[0, 0].set_title('Target distibution for tweets with missing location')

location_notna_target = df[df['location'].notna()]['target'].value_counts().rename({0:'Unreal disaster', 1:'Real disaster'})
location_notna_target.plot.bar(ax=ax[0, 1], color=['tab:blue', 'tab:orange'], rot=20)
ax[0, 1].set_title('Target distibution for tweets with non-missingg location')

df_location_unique = df[df['location'].isin(df['location'].value_counts()[df['location'].value_counts() == 1].index)]
location_unique = df_location_unique['target'].value_counts().rename({0:'Unreal disaster', 1:'Real disaster'})
location_unique.plot.bar(ax=ax[1, 0], color=['tab:blue', 'tab:orange'], rot=20)
ax[1, 0].set_title('Target distibution for tweets with unique location')

df_often_locations = df[df['location'].isin(df['location'].value_counts()[df['location'].value_counts() > 1].index)]
often_locations = df_often_locations['target'].value_counts().rename({0:'Unreal disaster', 1:'Real disaster'})
often_locations.plot.bar(ax=ax[1, 1], color=['tab:blue', 'tab:orange'], rot=20)
ax[1, 1].set_title('Target distibution for tweets with repeated location')

plt.show()



nltk.download('stopwords')
STOP_WORDS = stopwords.words("english")


def get_words(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub("\d+", " ", text)
    return [s for s in re.split("\W+", text) if len(s) > 0]


def extract_features_from_text(df):
    df['word_count'] = df['text'].apply(lambda x: len(get_words(x)))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(get_words(x))))
    df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in get_words(x) if w in STOP_WORDS]))
    df['average_word_length'] = df['text'].apply(lambda x: np.around(np.mean([len(w) for w in get_words(x)]), 1))
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['url_link_count'] = df['text'].apply(lambda x: len(re.findall('http[s]?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str(x))))
    df['hashtag_count'] = df['text'].apply(lambda x: len(re.findall('(^|\W)#', str(x))))
    df['mention_count'] = df['text'].apply(lambda x: len(re.findall('(^|\W)+@', str(x))))
    return df

df = extract_features_from_text(df)
df

df.to_csv('preprocessed-data.csv')

f,ax = plt.subplots(3, 3,figsize=(15,15))

FEATURES = [('word_count', 'Word count'), ('unique_word_count', 'Unique word count'), ('stop_word_count', 'Stop word count'),
            ('average_word_length', 'Average word length'), ('text_length', 'Text length'), ('punctuation_count', 'Punctuation character count'),
            ('url_link_count', 'URL link count'), ('hashtag_count', 'Hashtag count'), ('mention_count', 'Mention count')
           ]

for i, value in enumerate(FEATURES):
    feature, title = value
    
    df[df['target'] == 0][feature].hist(ax=ax[i // 3, i % 3], bins=20, alpha=0.5, label='Unreal disaster', color='red')
    df[df['target'] == 1][feature].hist(ax=ax[i // 3, i % 3], bins=20, alpha=0.5, label='Real disaster')
    ax[i // 3, i % 3].set_title(title)
    ax[i // 3, i % 3].set_ylabel('Frequency')
    ax[i // 3, i % 3].set_xlabel('Value')
    ax[i // 3, i % 3].legend(loc='upper right')

plt.show()


NORMALIZED_COLUMNS = list(df.columns[4:])

for column in NORMALIZED_COLUMNS:
    df[column] = (df[column] - df[column].mean()) / df[column].std()

df

def draw_heatmap(columns, corr_matrix):
    columns_len = len(columns)
    fig, ax = plt.subplots(figsize=(columns_len, columns_len))
    im = ax.imshow(corr_matrix, interpolation='nearest')
    fig.colorbar(im, orientation='vertical', fraction = 0.05)

    # Show all ticks and label them with the dataframe column name
    ax.set_xticks(range(columns_len))
    ax.set_yticks(range(columns_len))
    ax.set_xticklabels(columns, rotation=65, fontsize=15)
    ax.set_yticklabels(columns, rotation=0, fontsize=15)

    # Loop over data dimensions and create text annotations
    for i in range(columns_len):
        for j in range(columns_len):
            text = ax.text(j, i, round(corr_matrix[i, j], 2),
                        ha="center", va="center", color="black")

    plt.show()
    
def get_pearson_corr_matrix(columns: str):
    columns_count = len(columns)
    pearson_corr_matrix = np.zeros((columns_count, columns_count))
    for i1, column1 in enumerate(columns):
        for i2, column2 in enumerate(columns):
            pearson_corr_matrix[i1][i2] = df[column1].corr(df[column2], method='pearson')
    return pearson_corr_matrix


columns = NORMALIZED_COLUMNS + ['target']
pearson_corr_matrix = get_pearson_corr_matrix(columns)
draw_heatmap(columns, pearson_corr_matrix)



def get_embedding_path(embedding):
    embedding_zoo = {
        "crawl": "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec",
    }
    return embedding_zoo.get(embedding)

%%time
crawl_embeddings = KeyedVectors.load_word2vec_format(get_embedding_path('crawl'))



def get_vocab(X):
    vocab = {}
    texts = X.apply(lambda s: s.split()).values      
    for text in texts:
        for word in text:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1                
    return vocab

def check_embeddings_coverage(X, embeddings):
    vocab = get_vocab(X)
    covered, oov = {}, {} 
    n_covered, n_oov = 0, 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage
    
oov, vocab_coverage, text_coverage = check_embeddings_coverage(df['text'], crawl_embeddings)
print(f'Crawl Embeddings cover {vocab_coverage} of vocabulary and {text_coverage} of text')

def remove_url_links(text: str):
    return re.sub(r'http\S+', ' ', text)

def remove_shortened_forms(text: str):
    result = re.sub("won't", ' will not', text)
    result = re.sub("can't", ' can not', text)
    result = re.sub("n't", ' not', text)
    result = re.sub("'m", ' am', text)
    result = re.sub("'re", ' are', text)
    result = re.sub("'ve", ' have', text)
    result = re.sub("'ll", ' will', text)
    result = re.sub("'d", ' would', text)
    result = re.sub("'s", ' ', text)
    return result

def remove_stop_words(text: str):
    stop_words = stopwords.words("english")
    return ' '.join([w for w in text.split() if w not in stop_words])

def preprocessing(text):
    text = str(text).lower()
    text = remove_url_links(text)
    text = remove_shortened_forms(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = remove_stop_words(text)
    text = re.sub('\s+', ' ', text)
    return text
    
df['text'] = df['text'].apply(lambda x: preprocessing(x))

oov, vocab_coverage, text_coverage = check_embeddings_coverage(df['text'], crawl_embeddings)
print(f'Crawl Embeddings cover {vocab_coverage} of vocabulary and {text_coverage} of text')

def preprocessing_keywords(df):
    df['keyword'] = df['keyword'].fillna('empty')
    df['keyword'] = df['keyword'].apply(lambda x: x.replace('%20', ' '))
    return df['keyword']

df['keyword'] = preprocessing_keywords(df)



warnings.filterwarnings("ignore",  category = FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['target'], axis=1), df['target'], test_size=0.04, shuffle=True

%%capture
from transformers import TFBertModel, AutoTokenizer

bert = TFBertModel.from_pretrained('bert-large-uncased')
bert_text_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')





lstm_text_tokenizer = Tokenizer()
lstm_text_tokenizer.fit_on_texts(X_train['text'])

lstm_keyword_tokenizer = Tokenizer()
lstm_keyword_tokenizer.fit_on_texts(X_train['keyword'])

def prepare_lstm_model_inputs(X):
    seq = lstm_text_tokenizer.texts_to_sequences(X['text'])
    pad_seq = pad_sequences(seq)
    
    keywords_matrix = lstm_keyword_tokenizer.texts_to_matrix(X['keyword'])
    meta_features = X.loc[:, 'word_count':]
    float_features = np.concatenate([keywords_matrix, meta_features], axis=1)
    return pad_seq, float_features

train_pad_seq, train_float_features = prepare_lstm_model_inputs(X_train)
test_pad_seq, test_float_features = prepare_lstm_model_inputs(X_test)

print(train_pad_seq.shape)
print(train_float_features.shape)

train_pad_seq, train_float_features


WORD_EMBEDING_LENGTH = len(crawl_embeddings['word'])
VOCAB_SIZE = len(lstm_text_tokenizer.word_index) + 1
INPUT_LENGTH = len(train_pad_seq[0])
INPUT_FLOAT_FEATURES_LENGTH = len(train_float_features[0])

# Converting the words in our Vocabulary to their corresponding embeddings and placing them in a matrix.
embedding_matrix = np.zeros((VOCAB_SIZE, WORD_EMBEDING_LENGTH))
unknown_words = []
for word, i in lstm_text_tokenizer.word_index.items():
    try:
        embedding_vector = crawl_embeddings[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        unknown_words.append(word)

print(len(unknown_words))


logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



tf.get_logger().setLevel('INFO')


class LSTMModel:
    def __init__(self):
        self._model = self._build_model()

    @property
    def model(self) -> tf.keras.Model:
        return self._model
    
    def _build_model(self) -> tf.keras.Model:
        text_input = keras.Input(shape=(None,), name="text")
        x = Embedding(input_dim=VOCAB_SIZE, output_dim=WORD_EMBEDING_LENGTH, weights=[embedding_matrix], input_length=INPUT_LENGTH, trainable = False)(text_input)
        x = SpatialDropout1D(0.4)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = SpatialDropout1D(0.1)(x)
        text_features = LSTM(16)(x)
        
        float_features_input = keras.Input(shape=(INPUT_FLOAT_FEATURES_LENGTH,), name="float_features")
        x = Dense(64,activation = tf.keras.layers.Activation(tfa.activations.mish))(float_features_input)
        x = Dense(8,activation = tf.keras.layers.Activation(tfa.activations.mish))(x)
        meta_features = Dense(2,activation = tf.keras.layers.Activation(tfa.activations.mish))(x)
        
        features = concatenate([text_features, meta_features], axis=1)
        x = Dropout(0.2)(features)
        x = Dense(8, activation = tf.keras.layers.Activation(tfa.activations.mish))(x)
        x = Dense(4, activation = tf.keras.layers.Activation(tfa.activations.mish))(x)
        output = Dense(1,activation='sigmoid')(x)
        
        return keras.Model(inputs=[text_input, float_features_input], outputs=[output])
model = LSTMModel().model
model.summary()


keras.utils.plot_model(model, "model.png", show_shapes=True)

BATCH_SIZE = 1
EPOCHS_NUMBER = 20
VALIDATION_SPLIT = 0.1
STEPS_PER_EPOCH = int(len(train_pad_seq) // BATCH_SIZE * (1- VALIDATION_SPLIT))

def compile_model(model, learning_rate, warmup_proportion):
    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=learning_rate,
        total_steps=EPOCHS_NUMBER * STEPS_PER_EPOCH,
        warmup_proportion=warmup_proportion,
        min_lr=0.0000001,
    )
    optimizer = tfa.optimizers.Lookahead(optimizer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics = ['accuracy',
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall'),
                             tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5)]
                 )
    return model


def train_model(learning_rate, warmup_proportion):
    model = LSTMModel().model
    compile_model(model, learning_rate, warmup_proportion)

    checkpoint_filepath = f'/kaggle/working/checkpoints/lstm-model-checkpoint-{learning_rate}-{warmup_proportion}'
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True
    )


    model_history = model.fit({'text': train_pad_seq, 'float_features': train_float_features},
                              y_train,
                              epochs=EPOCHS_NUMBER,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_split=VALIDATION_SPLIT,
                              callbacks=[save_callback],
                              verbose=1)
    return model_history, checkpoint_filepath

def grid_search(params: list):
    results = []
    best_f1_score = 0
    best_checkpoint_filepath, best_model_history, best_params = None, None, None
    for model_params in params:
        learning_rate = model_params['learning_rate']
        warmup_proportion = model_params['warmup_proportion']
        model_history, checkpoint_filepath = train_model(learning_rate, warmup_proportion)
        f1_score = max(model_history.history['val_f1_score'])
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_checkpoint_filepath, best_model_history, best_params = checkpoint_filepath, model_history, model_params
        results.append({'params': model_params, 'model_history': model_history, 
                        'checkpoint_filepath': checkpoint_filepath, 'f1_score': f1_score,
                        'best_epoch_index': np.argmax(model_history.history['val_f1_score'])})
    
    print(results)
    print(f'The best params were found: {best_params}')
    
    return results, best_checkpoint_filepath, best_model_history, best_params


lstm_all_results, lstm_checkpoint_filepath, lstm_model_history, lstm_best_params = grid_search([
    {'learning_rate': 0.01, 'warmup_proportion': 0.1},
    {'learning_rate': 0.001, 'warmup_proportion': 0.2},
    {'learning_rate': 0.0005, 'warmup_proportion': 0.3},
    {'learning_rate': 0.0001, 'warmup_proportion': 0.3},
])


metrics_results = pd.DataFrame([[
    f"learning_rate = {res['params']['learning_rate']}, warmup_proportion = {res['params']['warmup_proportion']}",
    res['model_history'].history['val_accuracy'][res['best_epoch_index']],
    res['model_history'].history['val_precision'][res['best_epoch_index']],
    res['model_history'].history['val_recall'][res['best_epoch_index']],
    res['model_history'].history['val_f1_score'][res['best_epoch_index']],
] for res in lstm_all_results], columns=['params', 'accuracy', 'precision', 'recall', 'f1_score'])

fig, ax = plt.subplots()
metrics_results.plot.bar(ax=ax, x='params', figsize=(10, 8), rot=50)
ax.set_xlabel('params', fontsize=15)
ax.legend(prop={'size': 15})


def make_plot(loss, val_loss, acc, val_acc, precision, val_precision, recall, val_recall):
    t = np.arange(1,len(loss)+1,1)

    f, axs = plt.subplots(2, 2, figsize=(12,12))
    plt.subplots_adjust(wspace=0.2)

    axs[0,0].plot(t, loss)
    axs[0,0].plot(t, val_loss)
    axs[0,0].set_xlabel('epoch')
    axs[0,0].set_ylabel('loss')
    axs[0,0].set_title('Train vs Val loss')
    axs[0,0].legend(['train','val'], ncol=2, loc='upper right')

    axs[0,1].plot(t, acc)
    axs[0,1].plot(t, val_acc)
    axs[0,1].set_xlabel('epoch')
    axs[0,1].set_ylabel('acc')
    axs[0,1].set_title('Train vs Val acc')
    axs[0,1].legend(['train','val'], ncol=2, loc='upper right')

    axs[1,0].plot(t, precision)
    axs[1,0].plot(t, val_precision)
    axs[1,0].set_xlabel('epoch')
    axs[1,0].set_ylabel('precision')
    axs[1,0].set_title('Train vs Val precision')
    axs[1,0].legend(['train','val'], ncol=2, loc='upper right')
    
    axs[1,1].plot(t, recall)
    axs[1,1].plot(t, val_recall)
    axs[1,1].set_xlabel('epoch')
    axs[1,1].set_ylabel('recall')
    axs[1,1].set_title('Train vs Val recall')
    axs[1,1].legend(['train','val'], ncol=2, loc='upper right')

    plt.show()

loss = lstm_model_history.history['loss']
acc = lstm_model_history.history['accuracy']
precision = lstm_model_history.history['precision']
recall = lstm_model_history.history['recall']
val_loss = lstm_model_history.history['val_loss']
val_acc = lstm_model_history.history['val_accuracy']
val_precision = lstm_model_history.history['val_precision']
val_recall = lstm_model_history.history['val_recall']

make_plot(loss, val_loss, acc, val_acc, precision, val_precision, recall, val_recall)

model = LSTMModel().model
model.load_weights(lstm_checkpoint_filepath)
compile_model(model, lstm_best_params['learning_rate'], lstm_best_params['warmup_proportion'])

f1_score = lstm_model_history.history['f1_score']
val_f1_score = lstm_model_history.history['val_f1_score']

plt.figure()
plt.plot(lstm_model_history.epoch, f1_score, 'm', label='Training F1 Score')
plt.plot(lstm_model_history.epoch, val_f1_score, 'y', label='Validation F1 Score')

plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

lstm_loss, lstm_acc, lstm_precision, lstm_recall, lstm_f1_score = model.evaluate(
    {'text': test_pad_seq, 'float_features': test_float_features}, y_test)

print(f"\n loss = {lstm_loss} \n acc = {lstm_acc} \n precision = {lstm_precision} \n recall = {lstm_recall} \n f1_score = {lstm_f1_score}")