import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained model and tokenizer
model_path = "TRAINED_MODEL"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label names
label_keys = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Inference function
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    # Format results
    return {label: float(probs[idx]) for idx, label in enumerate(label_keys)}

# Define Gradio UI
iface = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter a comment..."),
    outputs=gr.Label(num_top_classes=6),
    title="Toxic Comment Classifier",
    description="Enter a comment and see the toxicity scores.",
)

# Run the app
if __name__ == "__main__":
    iface.launch(share=True)