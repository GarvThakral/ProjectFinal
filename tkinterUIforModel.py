import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
import re
import unicodedata
import logging

# Matplotlib imports for embedding charts in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from typing import Tuple, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

################################################################################
# Bengali Text Preprocessor (mirrors your training code)
################################################################################
class BengaliTextPreprocessor:
    """Bengali text preprocessor with cleaning and normalization."""
    
    def __init__(self):
        # Common Bengali stopwords (optional; not strictly needed for inference)
        self.bengali_stopwords = {
            'এই', 'ওই', 'সেই', 'কি', 'যে', 'কে', 'একটি', 'এর', 'কোন',
            'এবং', 'অথবা', 'কিন্তু', 'তাই', 'যদি', 'তবে', 'বা', 'থেকে'
        }
        
        # Bengali punctuations
        self.bengali_punctuations = set('।॥৷''"",.!?-:;')
        
        # Common Bengali character corrections
        self.char_maps = {
            '়': '়',
            'র্': 'র্',
            'য্': 'য্',
            '‌': '',  # Remove zero-width non-joiner
            '‍': ''   # Remove zero-width joiner
        }

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Apply character corrections
        for incorrect, correct in self.char_maps.items():
            text = text.replace(incorrect, correct)
        
        # Remove URLs and HTML
        text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


################################################################################
# BengaliNLIModelGUI: Loads the model from ./results and provides a Tkinter GUI
################################################################################
class BengaliNLIModelGUI:
    def __init__(self, master, model_dir: str = "./results"):
        """
        :param master: The Tkinter root or main window
        :param model_dir: Path to the directory where your model is saved
        """
        self.master = master
        self.master.title("Bengali NLI GUI")
        self.master.geometry("900x600")  # Adjust as needed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize text preprocessor
        self.preprocessor = BengaliTextPreprocessor()

        # Load tokenizer and model from the saved directory
        logger.info(f"Loading tokenizer and model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # ---------------------------------------------------------------------
        #  KEY CHANGE: Override config to ensure num_labels=3
        # ---------------------------------------------------------------------
        config = AutoConfig.from_pretrained(model_dir)
        config.num_labels = 3
        config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config
        )
        
        # Then wrap it with PEFT for inference if needed
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        # Create the UI elements
        self._create_widgets()

        # Prepare a Matplotlib figure for confidence bar chart
        self.fig, self.ax = plt.subplots(figsize=(5,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_widgets(self):
        """Create and place Tkinter widgets."""
        # Title
        title_label = ttk.Label(self.master, text="Multilingual NLI (Bengali) - LoRA Model",
                                font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Frame for input fields
        input_frame = ttk.Frame(self.master)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        # Premise
        premise_label = ttk.Label(input_frame, text="Premise:", font=("Helvetica", 12))
        premise_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.premise_entry = ttk.Entry(input_frame, width=70)
        self.premise_entry.grid(row=0, column=1, padx=5, pady=5)

        # Hypothesis
        hypothesis_label = ttk.Label(input_frame, text="Hypothesis:", font=("Helvetica", 12))
        hypothesis_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.hypothesis_entry = ttk.Entry(input_frame, width=70)
        self.hypothesis_entry.grid(row=1, column=1, padx=5, pady=5)

        # Predict button
        predict_button = ttk.Button(input_frame, text="Predict", command=self.on_predict_click)
        predict_button.grid(row=2, column=1, sticky=tk.E, padx=5, pady=5)

        # Prediction label
        self.prediction_label = ttk.Label(self.master, text="Prediction: N/A\nConfidence: N/A",
                                          font=("Helvetica", 12))
        self.prediction_label.pack(pady=10)

        # Frame for the chart
        self.chart_frame = ttk.Frame(self.master)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def on_predict_click(self):
        """When user clicks Predict, run the model on the premise-hypothesis pair."""
        premise = self.premise_entry.get()
        hypothesis = self.hypothesis_entry.get()

        # Preprocess and run inference
        predicted_label, confidence, distribution = self.predict_distribution(premise, hypothesis)

        # Update the prediction label
        self.prediction_label.config(
            text=f"Prediction: {predicted_label}\nConfidence: {confidence:.4f}"
        )

        # Plot the confidence distribution
        self.ax.clear()
        classes = [self.label_map[i] for i in range(len(self.label_map))]
        self.ax.bar(classes, distribution, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Prediction Confidence by Class")

        for i, val in enumerate(distribution):
            self.ax.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom')

        self.canvas.draw()

    def predict_distribution(self, premise: str, hypothesis: str) -> Tuple[str, float, List[float]]:
        """
        Returns (predicted_label, confidence, distribution) for the given premise-hypothesis pair.
        Distribution is a list of probabilities for each label [entailment, neutral, contradiction].
        """
        # Clean text
        clean_premise = self.preprocessor.preprocess(premise)
        clean_hypothesis = self.preprocessor.preprocess(hypothesis)

        # Tokenize
        inputs = self.tokenizer(
            clean_premise,
            clean_hypothesis,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=192  # Match your model's max length or use the same from training
        ).to(self.device)

        # Ensemble approach (like in your code)
        num_forward_passes = 5
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_forward_passes):
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probabilities.append(probabilities)

        mean_probabilities = torch.mean(torch.stack(all_probabilities), dim=0)
        predicted_class = torch.argmax(mean_probabilities, dim=-1).item()
        confidence = mean_probabilities[0][predicted_class].item()

        # Convert the distribution to a Python list
        distribution = mean_probabilities[0].tolist()

        return self.label_map[predicted_class], confidence, distribution


def main():
    root = tk.Tk()
    app = BengaliNLIModelGUI(root, model_dir="./results")
    root.mainloop()


if __name__ == "__main__":
    main()
