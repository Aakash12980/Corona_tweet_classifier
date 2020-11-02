from torch.utils.data import Dataset
from transformers import BertTokenizer
import pickle


class CoronaData(Dataset):

    def __init__(self, x, y=None):
        self.x = x
        if y is not None:
            self.y = y
        self.size = len(self.x)
    
    def __getitem__(self, index):
        if index < self.size:
            return self.x[index], self.y[index]
        else:
            raise("Index is larger than length of data")
    
    def __len__(self):
        return self.size

    @staticmethod
    def read_file(file_path):
        data = pickle.load(open(file_path, "rb"))
        return data["tweet"], data["label"]

   

