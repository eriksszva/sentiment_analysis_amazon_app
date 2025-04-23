# Amazon Review Sentiment Analysis using Transformer Embeddings and Deep Learning

This project presents a sentiment analysis built using transformer-based embeddings and deep learning models such as LSTM and GRU. The dataset consists of user reviews scraped from Amazon, which are classified into positive, neutral, and negative sentiments.

## Requirements

To run the project, you need the following libraries installed:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used is scraped from Amazon app in Google PlayStore. It contains product review texts without labels.

## Sentiment Annotation via Pretrained Transformer
Utilize the cardiffnlp/twitter-xlm-roberta-base-sentiment model to auto-label reviews based on sentiment. This pre-labeling approach is used because a dataset lacks explicit labels.

```python
pipeline('sentiment-analysis', model=model_name, tokenizer=model_name)
```

```bash
Analyzing Sentiment Using XLM Roberta Model: 100%|██████████| 500/500 [00:20<00:00, 24.75it/s]
```

## Steps

1. **Data Preprocessing**
   - Load the dataset and drop duplicates.
   - Clean the text data by removing unwanted characters such as HTML tags, URLs, and emojis.
   - Label encode the sentiment labels.
   - use DistilBERT tokenizer to convert reviews into numerical sequences (input_ids and attention_mask). This encoding preserves contextual information through transformer embeddings.

2. **Model Building**
   - **DistilBERT model**: Pretrained transformer-based model used for sentiment analysis.
   - **LSTM model**: A custom LSTM-based model built using TensorFlow Keras.
   - **GRU model**: A custom GRU-based model built using TensorFlow Keras.

3. **Model Training**
   - Train the models on the Amazon Reviews dataset with validation.
   - Apply early stopping to prevent overfitting.

4. **Evaluation**
   - Generate classification reports and confusion matrices for each model.
   - Visualize results using heatmaps.

5. **Inference**
   - Perform sentiment prediction on custom reviews.


### Model Summary

### First Schema: using LSTM and Tensorflow Embeddings
A simple yet effective LSTM model built from scratch using an Embedding layer followed by LSTM and Dense layers.

```python
Embedding(input_dim=vocab_size, output_dim=128) → LSTM(64) → Dense(32) → Dense(3)
```

```bash
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_ids (InputLayer)       [(None, 128)]             0         
_________________________________________________________________
embedding (Embedding)        (None, 128, 128)         3,906,816  
_________________________________________________________________
lstm (LSTM)                  (None, 64)               49,408    
_________________________________________________________________
dense (Dense)                (None, 32)               2,080      
_________________________________________________________________
dense_1 (Dense)              (None, 3)                99        
=================================================================
Total params: 11,875,211
Trainable params: 3,958,403
Non-trainable params: 0
_________________________________________________________________
```


### Training Logs (LSTM Model)

```bash
Epoch 1/100
750/750 [==============================] - 12s 11ms/step - accuracy: 0.3532 - loss: 1.0989 - val_accuracy: 0.6350 - val_loss: 0.9490
Epoch 2/100
750/750 [==============================] - 8s 11ms/step - accuracy: 0.7316 - loss: 0.8461 - val_accuracy: 0.7385 - val_loss: 0.6198
...
Epoch 21/100
750/750 [==============================] - 10s 11ms/step - accuracy: 0.9753 - loss: 0.0569 - val_accuracy: 0.8552 - val_loss: 0.7938
```


### Second Schema: using Distilbert Embeddings + LSTM (Frozen Embedding)
This hybrid model leverages DistilBERT as a frozen embedding extractor. The output from last_hidden_state is passed into a trainable LSTM layer, followed by Dense layers for classification.

```
bert_model(x).last_hidden_state → LSTM(64) → Dense(32) → Dense(3)
```

```bash
Model: "distilbert_lstm_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_ids (InputLayer)       [(None, 128)]             0         
bert_model (DistilBertModel) (None, 128, 768)          0  
lstm (LSTM)                  (None, 64)               213,248     
dense (Dense)                (None, 32)               2,080       
dense_1 (Dense)              (None, 3)                99         
=================================================================
Total params: 215,427
Trainable params: 215,427
Non-trainable params: 0
_________________________________________________________________
```

### LSTM Model Training Logs

```bash
Epoch 1/10
750/750 [==============================] - 139s 175ms/step - accuracy: 0.7794 - loss: 0.5740 - val_accuracy: 0.8248 - val_loss: 0.4310
Epoch 2/10
750/750 [==============================] - 143s 177ms/step - accuracy: 0.8372 - loss: 0.4167 - val_accuracy: 0.8300 - val_loss: 0.4285
...
Epoch 10/10
750/750 [==============================] - 142s 179ms/step - accuracy: 0.9177 - loss: 0.1845 - val_accuracy: 0.8447 - val_loss: 0.4196
```

### Third Schema: using Distilbert Embeddings + GRU (Frozen Embedding)

An alternative to LSTM using a GRU layer instead. GRU is typically faster and performs similarly on sequential data.
```
bert_model(x).last_hidden_state → GRU(64) → Dense(32) → Dense(3)
```

```bash
Model: "distilbert_gru_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_ids (InputLayer)       [(None, 128)]             0         
bert_model (DistilBertModel) (None, 128, 768)          0  
gru (GRU)                    (None, 64)               160,128      
dense (Dense)                (None, 32)               2,080       
dense_1 (Dense)              (None, 3)                99         
=================================================================
Total params: 162,307
Trainable params: 162,307
Non-trainable params: 0
_________________________________________________________________
```

### GRU Model Training Logs
```bash
Epoch 1/10
750/750 [==============================] - 156s 200ms/step - accuracy: 0.7830 - loss: 0.5592 - val_accuracy: 0.8253 - val_loss: 0.3962
Epoch 2/10
750/750 [==============================] - 184s 175ms/step - accuracy: 0.8488 - loss: 0.3793 - val_accuracy: 0.8248 - val_loss: 0.4008
...
Epoch 10/10
750/750 [==============================] - 142s 179ms/step - accuracy: 0.9401 - loss: 0.1402 - val_accuracy: 0.8505 - val_loss: 0.4842
```


## Conclusion

This project demonstrates the application of transformer models (DistilBERT) and custom RNN models (LSTM and GRU) for sentiment analysis. The results are visualized using confusion matrices and classification reports, providing insight into the model's performance. The custom reviews also receive predictions, allowing to see how the models classify new data.