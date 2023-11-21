import sys
from datasets import load_from_disk
from preprocess_dataset import make_t5_dataset
import pickle

dataset = load_from_disk('/home/surkov/workplace/custom_graphs/data/preprocessed_dataset')
print(dataset)
print(sys.argv)
dataset = {
    'train': dataset['train'].select(range(int(sys.argv[1]), int(sys.argv[2]))),
    'validation': dataset['validation'].select(range(int(sys.argv[3]), int(sys.argv[4]))),
}
result = make_t5_dataset(dataset)
with open(f'/home/surkov/workplace/custom_graphs/data/t5_dataset_{sys.argv[1]}-{sys.argv[2]}.pickle', 'wb') as f:
    pickle.dump(result, f)