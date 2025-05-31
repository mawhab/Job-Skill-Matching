import numpy as np
from torch.utils.data import Dataset
from common.preprocessing_utils import IdToName

class MultiNDataset(Dataset):
    def __init__(self, pairs, formatter):
        self.pairs = pairs
        self.formatter = formatter

    def __getitem__(self, idx):
        job_id, skill_ids = self.pairs[idx]
        job_alias = np.random.choice(IdToName.get_job_names(job_id))
        skill_id = np.random.choice(skill_ids)
        skill_name = np.random.choice(IdToName.get_skill_names(skill_id))
        
        item = {'job': self.formatter.format_job(job_alias), 'skill': self.formatter.format_skill(skill_name)}
        return item

    def __len__(self):
        return len(self.pairs)