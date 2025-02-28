import logging
import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import re
import unicodedata
from typing import Dict, Tuple, Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BengaliTextPreprocessor:
    """Bengali text preprocessor with cleaning and normalization."""
    
    def __init__(self):
        # Common Bengali stopwords
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

class BengaliNLIModel:
    """Resource-efficient Bengali NLI model using LoRA fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",  # Smaller base model
        max_length: int = 128,                 # Shorter sequence length
        num_labels: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = device
        self.preprocessor = BengaliTextPreprocessor()
        
        logger.info(f"Initializing Bengali NLI model with {model_name} on {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,                  # LoRA attention dimension
            lora_alpha=32,         # Alpha parameter for LoRA scaling
            lora_dropout=0.1,      # Dropout probability for LoRA layers
            target_modules=["query", "key", "value"]  # Apply LoRA to attention modules
        )
        
        # Create PEFT model
        self.model = get_peft_model(base_model, peft_config).to(device)
        
        # Label map
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def preprocess_dataset(self, examples: Dict) -> Dict:
        """Process dataset examples into model inputs."""
        # XNLI uses 'premise' and 'hypothesis' as column names
        if "premise" in examples and "hypothesis" in examples:
            sentence1 = [self.preprocessor.preprocess(text) for text in examples["premise"]]
            sentence2 = [self.preprocessor.preprocess(text) for text in examples["hypothesis"]]
        # Handle different column names if needed
        elif "sentence1" in examples and "sentence2" in examples:
            sentence1 = [self.preprocessor.preprocess(text) for text in examples["sentence1"]]
            sentence2 = [self.preprocessor.preprocess(text) for text in examples["sentence2"]]
        else:
            raise ValueError("Dataset format not recognized")
        
        tokenized = self.tokenizer(
            sentence1,
            sentence2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        if "label" in examples:
            tokenized["labels"] = examples["label"]
            
        return tokenized

    def prepare_datasets(
        self,
        train_size: int = 4000,    # Smaller training set
        val_size: int = 500,
        test_size: int = 500,
        seed: int = 42
    ) -> Dict:
        """Load and prepare datasets for training."""
        logger.info("Loading XNLI Bengali dataset...")
        
        try:
            # Load XNLI with Bengali language option
            dataset = load_dataset("xnli", "bn")
            logger.info("Successfully loaded XNLI Bengali dataset")
            
            # Get column names for debugging
            logger.info(f"Dataset columns: {dataset['train'].column_names}")
            
        except Exception as e:
            logger.error(f"Failed to load XNLI dataset: {str(e)}")
            
            # Fallback to other Bengali NLI dataset if available
            try:
                logger.info("Trying alternative Bengali NLI dataset...")
                dataset = load_dataset("csebuetnlp/xnli_bn")
                logger.info("Successfully loaded alternative Bengali dataset")
            except Exception as e2:
                logger.error(f"Failed to load alternative dataset: {str(e2)}")
                raise
        
        # Select smaller subsets for efficiency
        train_samples = dataset["train"].shuffle(seed=seed).select(range(min(train_size, len(dataset["train"]))))
        val_samples = dataset["validation"].shuffle(seed=seed).select(range(min(val_size, len(dataset["validation"]))))
        test_samples = dataset["test" if "test" in dataset else "validation"].shuffle(seed=seed).select(range(min(test_size, len(dataset["test" if "test" in dataset else "validation"]))))
        
        logger.info(f"Processing datasets: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
        
        # Process datasets with fewer workers to reduce memory usage
        train_dataset = train_samples.map(
            self.preprocess_dataset,
            batched=True,
            remove_columns=train_samples.column_names,
            num_proc=1  # Reduce parallel processing
        )
        
        val_dataset = val_samples.map(
            self.preprocess_dataset,
            batched=True,
            remove_columns=val_samples.column_names,
            num_proc=1
        )
        
        test_dataset = test_samples.map(
            self.preprocess_dataset,
            batched=True,
            remove_columns=test_samples.column_names,
            num_proc=1
        )
        
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }

    def train(
        self,
        output_dir: str = "./results",
        num_epochs: int = 5,           # Fewer epochs
        batch_size: int = 4,           # Smaller batch size
        learning_rate: float = 1e-4,   # Higher learning rate for LoRA
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        train_size: int = 4000         # Smaller training set by default
    ) -> None:
        """Train the model with resource-efficient configuration."""
        
        # Prepare datasets
        encoded_dataset = self.prepare_datasets(train_size=train_size)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            fp16=True if self.device == "cuda" else False,  # Mixed precision for faster training
            warmup_ratio=warmup_ratio,
            logging_dir="./logs",
            logging_steps=100,
            save_total_limit=1,  # Keep only the best model to save disk space
            dataloader_num_workers=1,  # Reduced for lower memory usage
            gradient_accumulation_steps=8,  # Accumulate gradients for effective larger batch
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        logger.info("Starting efficient training with LoRA for Bengali NLI...")
        trainer.train()
        
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate(encoded_dataset["test"])
        logger.info(f"Test set results: {eval_results}")
        
        self.save_model(output_dir)

    def predict(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[str, float]:
        """Make prediction for a single premise-hypothesis pair."""
        
        clean_premise = self.preprocessor.preprocess(premise)
        clean_hypothesis = self.preprocessor.preprocess(hypothesis)
        
        inputs = self.tokenizer(
            clean_premise,
            clean_hypothesis,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return self.label_map[predicted_class], confidence

    def save_model(self, output_dir: str) -> None:
        """Save the model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    @staticmethod
    def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted')
        
        acc = accuracy_score(labels, predictions)
        
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

def main():
    # Initialize Bengali NLI model with efficient configuration
    nli_model = BengaliNLIModel(
        model_name="xlm-roberta-base",  # Smaller base model
        max_length=128,                 # Shorter sequence length
    )
    
    # Train model with smaller dataset for Bengali
    nli_model.train(
        num_epochs=5,
        batch_size=4,
        train_size=4000
    )
    
    # Test cases for Bengali
    test_cases = [
        ("বাংলাদেশ একটি সুন্দর দেশ।", "বাংলাদেশের প্রাকৃতিক সৌন্দর্য অনন্য।"),
        ("তিনি প্রতিদিন সকালে ব্যায়াম করেন।", "তিনি স্বাস্থ্য সচেতন।"),
        ("আমি গতকাল সিনেমা দেখতে গিয়েছিলাম।", "আমি বাড়িতে ছিলাম।")
    ]
    
    logger.info("\nTesting Bengali NLI model with example cases:")
    for premise, hypothesis in test_cases:
        prediction, confidence = nli_model.predict(premise, hypothesis)
        logger.info(f"\nPremise: {premise}")
        logger.info(f"Hypothesis: {hypothesis}")
        logger.info(f"Prediction: {prediction} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()