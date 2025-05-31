import subprocess
import numpy as np
from common import defaults
from common.model_utils import get_embeddings
from sentence_transformers.util import cos_sim
from common.preprocessing_utils import ValidationData

def get_validation_mAP(encoder, tokenizer, formatter):
    queries_texts = ValidationData.get_queries_texts()
    corpus_texts = ValidationData.get_corpus_texts()
    jobs = [formatter.format_job(j) for j in queries_texts]
    skills = [formatter.format_skill(s) for s in corpus_texts]
    job_embeddings = get_embeddings(jobs, encoder=encoder, tokenizer=tokenizer, max_length=128, token='mean', batch_size=64)
    skill_embeddings = get_embeddings(skills, encoder=encoder, tokenizer=tokenizer, max_length=128, token='mean', batch_size=64)
    similarities = cos_sim(job_embeddings, skill_embeddings).cpu().numpy()
    return get_mAP(similarities)

def get_mAP(similarities, name=defaults.EXPERIMENT_NAME):
    results = []
    results_name = []
    
    corpus_ids = ValidationData.get_corpus_ids()
    map_queries = ValidationData.get_queries_ids_to_names()
    corpus_texts = ValidationData.get_corpus_texts()
    for q_idx, q_id in enumerate(ValidationData.get_queries_ids()):
        sorted_indices = np.argsort(-similarities[q_idx])
        used_doc_ids = set()
        rank_counter = 0
        for c_idx in sorted_indices:  # Consider the full list.
            doc_id = corpus_ids[c_idx]
            # If doc_id was already processed, go to the next one.
            if doc_id in used_doc_ids:
                continue
            used_doc_ids.add(doc_id)
            rank_counter += 1
    
            query_name = map_queries[q_id]
            doc_name = corpus_texts[c_idx]
            score = similarities[q_idx, c_idx]
    
            results.append(f"{q_id} Q0 {doc_id} {rank_counter} {score:.4f} {name}")
            results_name.append(f"{query_name} Q0 {doc_name} {rank_counter} {score:.4f} {name}")
    
    with open(defaults.RUN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    command = ["python", defaults.EVALUATION_SCRIPT, "--qrels", defaults.QRELS_FILE, "--run", defaults.RUN_FILE]
    result = subprocess.run(command, capture_output=True, text=True)

    return float(result.stdout.split('\n')[8].split(':')[1])