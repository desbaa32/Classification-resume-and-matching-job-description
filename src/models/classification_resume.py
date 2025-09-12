"""
Script de classification pour les CV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration des chemins
    data_path = 'data/processed/cleaned_combined_resume_final.csv'
    model_save_dir = Path('models/saved_model')
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    logger.info("Chargement des données CV")
    df = pd.read_csv(data_path)
    logger.info(f"Données chargées: {df.shape}")
    
    # Encodage des catégories
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])
    num_classes = len(le.classes_)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['Resume'], df['label'], test_size=0.2, random_state=42
    )
    
    # Tokenisation
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="tf"
    )
    
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="tf"
    )
    
    # Génération des embeddings
    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    
    def generate_embeddings_batch(encodings, batch_size=32):
        embeddings = []
        for i in range(0, encodings['input_ids'].shape[0], batch_size):
            batch = {
                'input_ids': encodings['input_ids'][i:i+batch_size],
                'attention_mask': encodings['attention_mask'][i:i+batch_size]
            }
            outputs = bert_model(batch)
            embeddings.append(outputs.last_hidden_state)
        return tf.concat(embeddings, axis=0)
    
    train_embeddings = generate_embeddings_batch(train_encodings)
    test_embeddings = generate_embeddings_batch(test_encodings)
    
    # Construction du modèle
    input_shape = train_embeddings.shape[1:]
    
    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(
        learning_rate=0.0005,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint(
            filepath=model_save_dir / 'best_cnn_model_resume_classifier.weights.h5',
            save_best_only=True,
            monitor='val_accuracy',
            save_weights_only=True
        )
    ]
    
    history = model.fit(
        train_embeddings,
        y_train_cat,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    test_loss, test_acc = model.evaluate(test_embeddings, y_test_cat)
    y_pred = model.predict(test_embeddings)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Rapport de classification
    report = classification_report(
        y_test, 
        y_pred_classes, 
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )
    
    # Sauvegarde des métriques
    metrics_data = []
    for class_name in le.classes_:
        metrics_data.append({
            'class_name': class_name,
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1_score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(model_save_dir / 'resume_class_metrics.csv', index=False)
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_, zero_division=0))
    
    # Sauvegarde du modèle
    model.save(model_save_dir / "CNN_final_model_classifier_resume.keras")
    joblib.dump(le, model_save_dir / 'label_encoder_resume.joblib')
    
    logger.info("Modèle sauvegardé avec succès")

if __name__ == "__main__":
    main()