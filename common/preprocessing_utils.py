import ast
import json
import numpy as np
import pandas as pd
from common import defaults
from common.model_utils import JobBERTEncoder
from sentence_transformers.util import cos_sim

def get_templates(encoder, descriptions=True):
    job_template = defaults.DESC_JOB_TEMPLATES[encoder] if descriptions else defaults.NO_DESC_JOB_TEMPLATES[encoder]
    skill_template = defaults.DESC_SKILL_TEMPLATES[encoder] if descriptions else defaults.NO_DESC_SKILL_TEMPLATES[encoder]
    
    if encoder == 'e5_instruct':
        if descriptions:
            job_template = job_template.format(task=defaults.DESC_E5_INSTRUCT_PROMPT, job='{job}', desc='{desc}')
        else:
            job_template = job_template.format(task=defaults.NO_DESC_E5_INSTRUCT_PROMPT, job='{job}', desc='{desc}')

    return job_template, skill_template

def get_train_df(path=defaults.TRAIN_DF):
    return pd.read_csv(path, sep='\t', names=['job', 'skill', 'category'])

def get_train_data():
    multindata_dict = {}
    df = get_train_df()

    for row in df.itertuples():
        job_data = multindata_dict.get(row.job, [])
        job_data.append(row.skill)
        multindata_dict[row.job] = job_data
    
    return [(k, v) for k,v in multindata_dict.items()]

def custom_collate_fn(batch, tokenizer, max_length):
    """
    Takes a list of dictionaries from JobSkillDataset.__getitem__.
    Tokenizes job and skill texts separately and pads them dynamically per batch.
    """
    job_texts = [item['job'] for item in batch]
    skill_texts = [item['skill'] for item in batch]

    # Tokenize jobs and skills separately
    tokenized_jobs = tokenizer(job_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    tokenized_skills = tokenizer(skill_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    return {
        'jobs': tokenized_jobs,
        'skills': tokenized_skills
    }

class ValidationData:
    queries = None
    corpus = None
    queries_texts = None
    corpus_texts = None
    queries_ids = None
    corpus_ids = None
    map_queries = None
    map_corpus = None

    @classmethod
    def initialize(cls, queries_file=defaults.VALIDATION_QUERIES, corpus_elements_file=defaults.VALIDATION_CORPUS_ELEMENTS):
        cls.queries = pd.read_csv(queries_file, sep='\t')
        corpus_elements = pd.read_csv(corpus_elements_file, sep='\t')
        corpus_elements["skill_aliases"] = corpus_elements["skill_aliases"].apply(lambda x: ast.literal_eval(x))
        cls.corpus = corpus_elements.explode("skill_aliases")

        cls.queries_texts = cls.queries.jobtitle.to_list()
        cls.corpus_texts = cls.corpus.skill_aliases.to_list()
        cls.queries_ids = cls.queries.q_id.to_list()
        cls.corpus_ids = cls.corpus.c_id.to_list()
        cls.map_queries = dict(zip(cls.queries.q_id.to_list(), cls.queries.jobtitle.to_list()))
        cls.map_corpus = dict(zip(cls.corpus.skill_aliases.to_list(), cls.corpus.c_id.to_list()))

    @classmethod
    def get_queries_texts(cls):
        if cls.queries is None:
            cls.initialize()
        return cls.queries

    @classmethod
    def get_corpus_texts(cls):  
        if cls.corpus is None:
            cls.initialize()
        return cls.corpus
    
    @classmethod
    def get_queries_ids(cls):
        if cls.queries is None:
            cls.initialize()
        return cls.queries_ids  
    
    @classmethod
    def get_corpus_ids(cls):
        if cls.corpus is None:
            cls.initialize()
        return cls.corpus_ids
    
    @classmethod
    def get_queries_ids_to_names(cls):
        if cls.queries is None:
            cls.initialize()
        return cls.map_queries
    
    @classmethod
    def get_corpus_ids_to_names(cls):
        if cls.corpus is None:
            cls.initialize()
        return cls.map_corpus
    

class ValTrainJobMatcher:
    query_embedding = None
    all_job_embedding = None
    val_jobs = None
    sim_matrix = None

    @classmethod
    def initialize(cls):
        cls.val_jobs = ValidationData.get_queries_texts()
        cls.train_jobs = IdToName.get_all_job_names()
        cls.train_embedding = JobBERTEncoder.encode(cls.train_jobs)
        cls.val_embedding = JobBERTEncoder.encode(cls.val_jobs)
        cls.sim_matrix = cos_sim(cls.val_embedding, cls.train_embedding)

    @classmethod
    def get_job_matches(cls, job, n=1, cutoff=0.97):
        job_index = cls.val_jobs.index(job)
        sim = cls.sim_matrix[job_index].squeeze()
        matches = np.argsort(-sim)[:n]
        return [cls.train_jobs[i] for i in matches if sim[i] > cutoff]
    
class InputFormatter:
    job_template = None
    skill_template = None
    job_descriptions = None
    skill_descriptions = None

    @classmethod
    def initialize(cls, job_template, skill_template):
        cls.job_template = job_template
        cls.skill_template = skill_template
        
        cls.load_job_descriptions()
        
        cls.skill_descriptions = get_skill_descriptions()

    @classmethod
    def load_job_descriptions(cls):
        pass

    @classmethod
    def format_job(cls, job):
        desc = cls.job_descriptions[job]
        return cls.job_template.format(job=job, desc=desc)

    @classmethod
    def format_skill(cls, skill):
        desc = cls.skill_descriptions[skill]
        return cls.skill_template.format(skill=skill, desc=desc)
    
    @classmethod
    def set_context(cls, context):
        pass

class NoDescInputFormatter(InputFormatter):

    @classmethod
    def format_job(cls, job):
        return cls.job_template.format(job=job)
    
    @classmethod
    def format_skill(cls, skill):
        return cls.skill_template.format(skill=skill)
    

class ESCOInputFormatter(InputFormatter):
    context = None

    @classmethod
    def load_job_descriptions(cls):
        job_name_desc = {}
        df = pd.read_csv(defaults.ESCO_JOBS_DF)
        for row in df.itertuples():
            for alias in row.aliases.split('\n'):
                job_name_desc[alias] = row.description
    
    @classmethod
    def format_job(cls, job):
        if cls.context == 'val':
            matches = ValTrainJobMatcher.get_job_matches(job)
            if matches:
                closest_job = matches[0]
                desc = cls.job_descriptions[closest_job]
            else:
                return cls.job_template.split(' [SEP]')[0].format(job=job)
        else:
            desc = cls.job_descriptions[job]
        return cls.job_template.format(job=job, desc=desc)

    @classmethod
    def set_context(cls, context):
        cls.context = context


class LLMInputFormatter(InputFormatter):
    all_job_descriptions = None

    @classmethod
    def load_job_descriptions(cls):
        with open(defaults.LLM_GENERATED_TRAIN_DESCS, 'r') as f:
            train_job_descriptions = json.load(f)
        
        with open(defaults.LLM_GENERATED_VAL_DESCS, 'r') as f:
            val_job_descriptions = json.load(f)
        
        cls.all_job_descriptions = {'train': train_job_descriptions, 'val': val_job_descriptions}
        cls.job_descriptions = cls.all_job_descriptions['train']

    @classmethod
    def set_context(cls, context):
        cls.job_descriptions = cls.all_job_descriptions[context]

def get_formatter(augmentation):
    if augmentation == 'esco':
        return ESCOInputFormatter
    elif augmentation == 'llm':
        return LLMInputFormatter
    else:
        return NoDescInputFormatter

class IdToName:
    jobid2name = None
    skillid2name = None

    @classmethod
    def initialize(cls):
        cls.get_jobid2name()
        cls.get_skillid2name()

    @classmethod
    def get_jobid2name(cls, path=defaults.JOBID2NAME):
        with open(path, 'r') as f:
            cls.jobid2name = json.load(f)

    @classmethod
    def get_skillid2name(cls, path=defaults.SKILLID2NAME):
        with open(path, 'r') as f:
            cls.skillid2name = json.load(f)

    @classmethod
    def get_job_names(cls, jobid):
        if cls.jobid2name is None:
            cls.initialize()
        return cls.jobid2name[jobid]
    
    @classmethod
    def get_all_job_names(cls):
        if cls.jobid2name is None:
            cls.initialize()
        return [alias for aliases in cls.jobid2name.values() for alias in aliases]
    
    @classmethod
    def get_skill_names(cls, skillid):
        if cls.skillid2name is None:
            cls.initialize()
        return cls.skillid2name[skillid]

def get_skill_descriptions():
    skill_link_desc = {}
    for row in pd.read_csv(defaults.ESCO_SKILLS_DF).itertuples():
        if type(row.altLabels) is float:
            skill_link_desc[row.preferredLabel] = row.description
        else:
            for alias in [row.preferredLabel] + row.altLabels.split('\n'):
                skill_link_desc[alias] = row.description
    return skill_link_desc