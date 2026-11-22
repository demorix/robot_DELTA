from transformers import pipeline
import torch
import gc

class IntentClassifier:
    def __init__(self, model_path="./intent_model"):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            device=-1
        )

    
    def classify(self, text: str):
        result = self.classifier(text)
        
        label = result[0]["label"]
        
        return label


    def close(self):
        del self.classifier
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()