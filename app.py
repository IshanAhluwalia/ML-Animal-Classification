from fastai.vision.all import *
import gradio as gr

# Load the trained model
learn = load_learner("animal_classifier.pkl")
labels = learn.dls.vocab

def classify(img):
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🐾 Animal Species Classifier",
    description="Upload an animal photo – the model returns its top guesses."
)

demo.launch()






