from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov import build_model
from datasets import load_dataset
import re
import pickle
import random
from tqdm import tqdm

dataset = load_dataset("grail_qa")

with open('/home/surkov/workplace/custom_graphs/data/entities_dict.pickle', 'rb') as f:
    ent_dict = pickle.load(f)

with open('/home/surkov/workplace/custom_graphs/data/relations_dict.pickle', 'rb') as f:
    rel_dict = pickle.load(f)

rels_list = list(rel_dict.keys())

def modify_sparql(example):

    def alter_s_expression(s_exp):
        def replacer(match):
            if match.group() not in ent_dict:
                return '[!!!]'
            return '['+ent_dict[match.group()]+']'
        return re.sub('m\.0[a-z0-9]*', replacer, s_exp)

    s_exp = example['s_expression']
    s_exp = alter_s_expression(s_exp)
    example['s_expression_mod'] = s_exp
    return example


def filter_bad_questions(example):
    return '[!!!]' not in example['s_expression']


def get_relevant_unrelevant_rels(example):
    s_exp = example['s_expression']
    rels_iter = re.finditer(r'([a-z\_]+\.)+[a-z\_]+', s_exp)
    rels = []
    for match in rels_iter:
        rel = match.string[match.span()[0]:match.span()[1]]
        rels.append(rel)
        if rel not in rel_dict.keys():
            example['s_expression'] += '[!!!]'
            break
    
    bad_rels = []
    for _ in range(len(rels)):
        while True:
            bad_rel = random.choice(rels_list)
            if bad_rel not in rels:
                bad_rels.append(bad_rel)
                break
    
    example['good_rels'] = rels
    example['bad_rels'] = bad_rels
    return example


def make_relations_ranking_dataset(dataset):
    result = {
        'train': [],
        'valid': []
    }
    for split in ['train', 'validation']:
        nsplit = split if split == 'train' else 'valid'
        for example in dataset[split]:
            question = example['question']
            for rel in example['good_rels']:
                result[nsplit].append([[question, [rel_dict[rel]]], '1'])
            for rel in example['bad_rels']:
                result[nsplit].append([[question, [rel_dict[rel]]], '0'])
    random.shuffle(result['train'])
    random.shuffle(result['valid'])
    result['test'] = result['valid']

    return result


def make_t5_dataset(dataset):
    result = {
        'train': [],
        'valid': []
    }

    rel_ranker = RelRankerInfer(
        load_path='/home/surkov/workplace/custom_graphs/data',
        rel_q2name_filename='relations_dict_new.pickle',
        ranker=build_model(
            "configs/relation_ranking_infer.json"
        )
    )

    for split in ['train', 'validation']:
        nsplit = split if split == 'train' else 'valid'
        for example in tqdm(dataset[split]):
            question = example['question']
            candidate_rels = list(rel_ranker.rel_q2name)
            probas = rel_ranker.rank_rels(question, candidate_rels)
            relevant_rels = []
            for i in range(10):
                relevant_rels.append(probas[i][0])
            s_expression = example['s_expression_mod']
            for rel in sorted(example['good_rels'], key=lambda x: -len(x)):
                s_expression = re.sub(f'([^\[]){rel}([^\]])', f'\\1[{rel}]\\2', s_expression)
            result[nsplit].append([[question, relevant_rels], s_expression])
            print(result[nsplit][-1])
    result['test'] = result['valid']
    return result
