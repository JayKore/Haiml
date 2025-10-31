import spacy
from spacy.matcher import PhraseMatcher

# Load a pre-trained spaCy model (download if missing)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Create a PhraseMatcher and add medical term patterns (case-insensitive)
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
medical_terms = [
    "fever", "cough", "headache", "diabetes", "hypertension", "pneumonia",
    "paracetamol", "amoxicillin", "surgery", "mri", "blood test", "patient",
    "male", "female", "age", "doctor", "nurse", "hospital"
]
patterns = [nlp.make_doc(text) for text in medical_terms]
matcher.add("MEDICAL_ENTITY", patterns)

def medical_entity_extraction(text):
    doc = nlp(text)
    entities = []

    # Collect spaCy's builtin NER results
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    # Use PhraseMatcher for the medical terms
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        entities.append({"text": span.text, "label": "MEDICAL_TERM"})

    return entities

# Example medical report
medical_report = """
Patient Name: John Doe
Date: 2025-09-10
Report: The patient, a 55-year-old male, presented with symptoms including a persistent cough and
high fever for three days. He also reported a severe headache. Past medical history includes
hypertension. A chest X-ray was performed, and a blood test was ordered. The doctor prescribed
Amoxicillin and advised rest. Follow-up visit scheduled.
"""

extracted_entities = medical_entity_extraction(medical_report)
print("Extracted Entities:")
for entity in extracted_entities:
    print(f" Text: {entity['text']}, Label: {entity['label']}")

# after extracting
print("Debug: spaCy model:", nlp.meta.get("name"))
print("Full text length:", len(medical_report))

doc = nlp(medical_report)
print("spaCy builtin NER entities (count):", len(doc.ents))
for ent in doc.ents:
    print(f"  - ENT: '{ent.text}' | label: {ent.label_} | span: ({ent.start_char}, {ent.end_char})")

# PhraseMatcher matches
matches = matcher(doc)
print("PhraseMatcher matches (count):", len(matches))
for match_id, start, end in matches:
    span = doc[start:end]
    rule_name = nlp.vocab.strings[match_id]
    print(f"  - MATCH: '{span.text}' | rule: {rule_name} | span: ({span.start_char}, {span.end_char})")

# If you still want the combined function output
extracted_entities = medical_entity_extraction(medical_report)
print("Combined extracted entities (count):", len(extracted_entities))
for e in extracted_entities:
    print(" ", e)