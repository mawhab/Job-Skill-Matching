import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

def get_model(model_name, device):
    if model_name=='escoxlm_r':
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        encoder = AutoModel.from_pretrained("jjzha/esco-xlm-roberta-large").to(device)
    elif model_name=='e5_large':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        encoder = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
    elif model_name=='e5_instruct':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        encoder = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').to(device)
    
    return encoder, tokenizer

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def calculate_mnrl_loss(job_embeds, skill_embeds, temperature):
    job_embeds = F.normalize(job_embeds, p=2, dim=1)
    skill_embeds = F.normalize(skill_embeds, p=2, dim=1)

    similarity_matrix = torch.matmul(job_embeds, skill_embeds.T) # (batch_size, batch_size)

    similarity_matrix = similarity_matrix / temperature

    batch_size = job_embeds.size(0)
    labels = torch.arange(batch_size, dtype=torch.long, device=job_embeds.device)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

class JobBERTEncoder:
    jobbert = None

    @classmethod
    def initialize(cls, device='cuda'):
        cls.jobbert = SentenceTransformer("TechWolf/JobBERT-v2").to(device)
    
    @classmethod
    def encode_batch(cls, texts):
        features = cls.jobbert.tokenize(texts)
        features = batch_to_device(features, cls.jobbert.device)
        features["text_keys"] = ["anchor"]
        with torch.no_grad():
            out_features = cls.jobbert.forward(features)
        return out_features["sentence_embedding"].cpu().numpy()

    @classmethod
    def encode(cls, texts, batch_size = 1024):
        if cls.jobbert is None:
            cls.initialize()

        sorted_indices = np.argsort([len(text) for text in texts])
        sorted_texts = [texts[i] for i in sorted_indices]
        
        embeddings = []
        
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i:i+batch_size]
            embeddings.append(cls.encode_batch(batch))
        
        sorted_embeddings = np.concatenate(embeddings)
        original_order = np.argsort(sorted_indices)
        return sorted_embeddings[original_order]
    
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(texts, encoder, tokenizer, batch_size=256, max_length=64, token='mean', device='cuda'):
    text_embeddings = torch.tensor([], device=device)
    for i in range(0, len(texts), batch_size):
        tokenized = {k:v.to(device) for k,v in tokenizer(texts[i:i+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').items()}
        with torch.no_grad():
            outputs = encoder(**tokenized)
        if token=='mean':
            embeddings = average_pool(outputs.last_hidden_state, tokenized['attention_mask'])
        else:
            embeddings = outputs.last_hidden_state[:,0,:]
        text_embeddings = torch.cat((text_embeddings, embeddings), dim=0)
    return text_embeddings