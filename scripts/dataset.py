from torch.utils.data import Dataset

# scitail datasets
class SciTailDataset(Dataset):
    def __init__(self, file : str):
        self.samples = []

        label_encodings = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        with open(file, 'r', encoding='utf8') as input_data:
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                label = line[0]
                premise = line[5]
                hypothesis = line[6]

                self.samples.append({'claim': hypothesis,'rationale':premise, 'label':label_encodings[label]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
