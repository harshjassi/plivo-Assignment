
LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]

# compute maps
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for lab, i in LABEL_TO_ID.items()}

# PII entity types (without B-/I- prefixes)
PII_TYPES = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}

def is_pii_label(label_name: str) -> bool:
    """Return True if label_name corresponds to a PII entity type."""
    # label_name is like "CREDIT_CARD" or "PHONE"
    return label_name in PII_TYPES
