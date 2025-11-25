from transformers import AutoModelForTokenClassification
from labels import LABEL_TO_ID

def build_token_classifier(base_model_name: str):
    """
    Returns an AutoModelForTokenClassification initialized with the label maps.
    """
    num_labels = len(LABEL_TO_ID)
    
    # Build id2label mapping fresh (to avoid direct import from labels)
    id2label = {v: k for k, v in LABEL_TO_ID.items()}
    
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=LABEL_TO_ID,
    )
    return model
