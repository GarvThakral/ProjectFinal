# ---------------------------------------------------------------------
#  KEY CHANGE: Create config manually with model_type since auto-detection failed
# ---------------------------------------------------------------------
try:
    config = AutoConfig.from_pretrained(model_dir)
except ValueError:
    # Create a default config with xlm-roberta as the base model type
    # (or whatever model type you're actually using)
    config = AutoConfig.from_pretrained("xlm-roberta-base")
    logger.info(f"Using default xlm-roberta configuration since model_type wasn't detected")

# Override config settings for NLI task
config.num_labels = 3
config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

# You may also need to specify the model_type explicitly 
config.model_type = "xlm-roberta"

# Then load the model with this config
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    config=config,
    ignore_mismatched_sizes=True  # Add this to handle potential size mismatches
)
