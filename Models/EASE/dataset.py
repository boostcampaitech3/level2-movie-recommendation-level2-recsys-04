from scipy.sparse import csr_matrix

from torch.utils.data import Dataset


class NEASEDataset(Dataset):
    def __init__(self, df, user_num, item_num):
        self.user_num = user_num
        self.item_num = item_num
        self.length = len(df)

        self.make_matrix()

    def make_matrix(self):
        values = [1.0] * self.length
        self.matrix = csr_matrix((values, (self.user_num, self.item_num))).todense()

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        return self.matrix[idx]