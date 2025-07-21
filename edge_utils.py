from presidio_analyzer import AnalyzerEngine
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import requests
import Two_layer_ldp

class EdgeProcessor:
    def __init__(self,
                 model_name="microsoft/phi-3.5-mini",
                 risk_threshold=0.5,
                 delta_threshold=0.1,
                 device="cuda"):
        self.analyzer = AnalyzerEngine()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.risk_threshold = risk_threshold
        self.delta_threshold = delta_threshold

        # 示例 domain
        self.type_domain = ["PERSON", "EMAIL_ADDRESS", "LOCATION", "PHONE_NUMBER"]
        self.value_domain_dict = {
            "PERSON": ["Alice", "Bob", "Charlie"],
            "LOCATION": ["Paris", "Tokyo", "New York"],
            "EMAIL_ADDRESS": ["user@example.com", "test@test.com"],
            "PHONE_NUMBER": ["123-456-7890", "555-555-5555"]
        }

    def run_ner(self, text):
        results = self.analyzer.analyze(text=text, language="en")
        return [(r.entity_type, r.start, r.end, text[r.start:r.end]) for r in results]

    def compute_risk_score(self, entities):
        risk_weights = {"PERSON": 1.0, "EMAIL_ADDRESS": 1.0, "LOCATION": 0.8, "PHONE_NUMBER": 0.9}
        score = sum(risk_weights.get(t, 0.2) for t, _, _, _ in entities)
        return score / max(len(entities), 1)

    def compute_cross_entropy(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs.input_ids[:, 1:]
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), reduction='mean')
        return loss.item()

    def cross_entropy_diff(self, prompt, entity_text):
        baseline_loss = self.compute_cross_entropy(prompt)
        masked_text = prompt.replace(entity_text, "[MASK]")
        masked_loss = self.compute_cross_entropy(masked_text)
        return masked_loss - baseline_loss

    def encrypt_sensitive_entities(self, prompt, entities):
        protected_prompt = prompt
        for entity_type, _, _, entity_text in entities:
            delta = self.cross_entropy_diff(prompt, entity_text)
            if delta > self.delta_threshold:
                noisy_result = Two_layer_ldp.two_layer_dp(entity_type, entity_text,
                                             self.type_domain, self.value_domain_dict,
                                             epsilon1=1.0, epsilon2=0.8)
                protected_prompt = protected_prompt.replace(entity_text, noisy_result)
        return protected_prompt

    def extract_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        embeddings = self.model.get_input_embeddings()(inputs.input_ids)
        return embeddings.detach().cpu().tolist()

    def send_to_cloud(self, embedding):
        resp = requests.post("http://127.0.0.1:8000/sketch", json={"embedding": embedding})
        return resp.json().get("sketch", "")
