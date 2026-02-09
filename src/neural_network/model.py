import os

# --- FIX CRITIC PENTRU COMPATIBILITATE (Trebuie să fie prima linie) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification


def build_model(
    *,
    model_name: str = "distilbert-base-multilingual-cased",
    learning_rate: float = 3e-5,
):
    """
    Model NLP (DistilBERT multilingual) pentru clasificare binară.

    Output: un singur logit (num_labels=1).
    - În training folosim BinaryCrossentropy(from_logits=True).
    - În inferență probabilitatea se obține cu sigmoid(logit).
    """
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        use_safetensors=False,  # important în unele medii TF
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
