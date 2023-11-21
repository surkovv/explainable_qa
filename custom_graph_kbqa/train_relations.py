from deeppavlov import build_model, train_model
from models.kbqa.rel_ranking_infer import RelRankerInfer
from datasets import load_from_disk
import pickle
from tqdm import tqdm

dataset_big = load_from_disk('/home/surkov/workplace/custom_graphs/data/preprocessed_dataset')['train']

def train_one():
    train_model('/home/surkov/workplace/custom_graphs/explainable_qa/custom_graph_kbqa/configs/rel_ranking_roberta_en.json')

def make_dataset(N):
    model = build_model('test.json')
    questions_entries = dataset_big.shuffle().select(range(N))
    questions = [questions_entries[i]['question'] for i in range(N)]
    relevant = [questions_entries[i]['good_rels'] for i in range(N)]
    ids_batch, names_batch = model(questions)

    dataset = []
    for question, rels, ids, names in zip(questions, relevant, ids_batch, names_batch):
        for rel in rels:
            dataset.append([[question, [rel]], '1'])
        for rel in ids[:5]:
            if rel not in rels:
                dataset.append([[question, [rel]], '0'])

    with open('/home/surkov/workplace/custom_graphs/data/rel_rank_dataset_1000.pickle', 'wb') as f:
        pickle.dump({
            'train': dataset,
            'valid': dataset,
            'test': dataset
        }, f)

for i in range(100):
    make_dataset(1000)
    train_one()