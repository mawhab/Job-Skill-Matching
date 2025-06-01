TRAIN_DF = 'data/task_data/training/job2skill.tsv'
JOBID2NAME = 'data/task_data/training/jobid2terms.json'
SKILLID2NAME = 'data/task_data/training/skillid2terms.json'

ESCO_JOBS_DF = 'data/esco_data/occupations_en.csv'
ESCO_SKILLS_DF = 'data/esco_data/skills_en.csv'

VALIDATION_QUERIES = 'data/task_data/validation/queries'
VALIDATION_CORPUS_ELEMENTS = 'data/task_data/validation/corpus_elements'

LLM_GENERATED_TRAIN_DESCS = 'data/generated_data/train_descriptions.json'
LLM_GENERATED_VAL_DESCS = 'data/generated_data/val_descriptions.json'
# LLM_GENERATED_VAL_DESCS = 'data/generated_data/test_descriptions.json'

NO_DESC_JOB_TEMPLATES = {
    'escoxlm_r': 'Job: {job}',
    'e5_large': 'query: Job: {job}',
    'e5_instruct': 'Instruct: {task}\nQuery: Job: {job}'
}

NO_DESC_SKILL_TEMPLATES = {
    'escoxlm_r': 'Skill: {skill}',
    'e5_large': 'passage: Skill: {skill}',
    'e5_instruct': 'Skill: {skill}'
}

DESC_JOB_TEMPLATES = {
    'escoxlm_r': 'Job: {job} [SEP] Description: {desc}',
    'e5_large': 'query: Job: {job} [SEP] Description: {desc}',
    'e5_instruct': 'Instruct: {task}\nQuery: Job: {job} [SEP] Description: {desc}'
}

DESC_SKILL_TEMPLATES = {
    'escoxlm_r': 'Skill: {skill} [SEP] Description: {desc}',
    'e5_large': 'passage: Skill: {skill} [SEP] Description: {desc}',
    'e5_instruct': 'Skill: {skill} [SEP] Description: {desc}'
}

DESC_E5_INSTRUCT_PROMPT = 'Given a job title and its description, retrieve relevant skills based on their descriptions.'
NO_DESC_E5_INSTRUCT_PROMPT = 'Given a job title, retrieve relevant skills.'

EVALUATION_SCRIPT = 'talentclef25_evaluation_script/talentclef_evaluate.py'
QRELS_FILE = 'data/task_data/validation/qrels.tsv'
RUN_FILE = 'data/evaluation/validation_pred.trec'

EXPERIMENT_NAME = 'Team_B_exp'

LLM_SYSTEM_PROMPT = """You are an expert HR assistant specialized in writing job descriptions. \
Generate a highly concise, professional, and factual job description based ONLY on the provided job title. \
Focus on typical key responsibilities. **Crucially, explicitly list 2-3 common, essential skills associated with the role within the description, often towards the end.** \
Do not add information not directly implied by the job title. \
The description must be brief, ideally 2-3 sentences maximum, to fit within processing limits. \
Output only the description text, without any preamble or introductory phrases."""