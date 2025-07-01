# src/model_handler.py

"""
Manages pre-trained models for translation and embedding extraction. [cite: 7]
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from typing import List
import numpy as np

class ModelHandler:
    """
    A class to handle model loading, translation, and embedding extraction.
    """
    def __init__(self, embedding_model_name: str = 'xlm-roberta-base', device: str = 'cpu'):
        """
        Initializes the handler by loading the multilingual embedding model.

        Args:
            embedding_model_name (str): Name of the Hugging Face model for embeddings. [cite: 7]
            device (str): The device to run models on ('cpu' or 'cuda').
        """
        self.device = device
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.translation_models = {}

    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """

        Generates sentence embeddings for a list of sentences using masked mean pooling.

        """
        self.embedding_model.eval()
        inputs = self.embedding_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        masked_embeddings = outputs.last_hidden_state * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        embeddings = summed / counts
        return embeddings.cpu().numpy()
    
    def zero_shot_translate(self, sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Performs zero-shot translation using mBART50 model.
        """
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        tokenizer.src_lang = src_lang
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate(self, sentences: List[str], model_name: str) -> List[str]:
        """
        Performs bidirectional translation using a specified model. 

        Args:
            sentences (List[str]): Sentences to translate.
            model_name (str): Hugging Face name of the translation model.

        Returns:
            List[str]: A list of translated sentences.
        """
        if model_name not in self.translation_models:
            print(f"Loading translation model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.translation_models[model_name] = (tokenizer, model)

        tokenizer, model = self.translation_models[model_name]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        generated_tokens = model.generate(**inputs)
        translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated_texts