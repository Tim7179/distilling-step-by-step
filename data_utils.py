# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datasets import Dataset, DatasetDict, load_dataset


DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class CQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'cqa'
        source_dataset_name = 'cos_e'
        dataset_version = 'v1.11'
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'validation',
        }
        batch_size = 1000
        train_batch_idxs = range(10)
        test_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]
            c_4 = example['choices'][4]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['id', 'question', 'choices', 'answer', 'abstractive_explanation', 'extractive_explanation'])

        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('So the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
            label = label if label[-1]!='.' else label[:-1]
        except:
            label = ' '
        
        return rationale, label


    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('So the answer is')
            rationale = rationale.rstrip()
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        try:
            label = re.split(r'\(.\)', label)[1].strip()
            label = label if label[-1]!='.' else label[:-1]
        except:
            label = ' '
        
        return rationale, label

class OpenR1Math220kDatasetLoader(DatasetLoader):
    def __init__(self, test_size: float = 0.05, seed: int = 42, chunk_size: int = 500):
        dataset_name         = 'OpenR1-Math-220k'
        source_dataset_name  = 'open-r1/OpenR1-Math-220k'
        dataset_version      = 'all'
        has_valid            = False
        split_map            = {'train': 'train', 'test': 'test'}
        batch_size           = chunk_size   # 與 data_transfer.py 的 CHUNK_SIZE 保持一致

        # 一次性載入並做格式轉換
        raw_ds   = load_dataset(source_dataset_name, dataset_version, split='train')
        def _fmt(ex):
            return {
                # 'id'   : ex['uuid'],
                'input': ex['problem'],
                'label': f"{ex['solution']}\nAnswer: {ex['answer']}"
            }
        proc_ds  = raw_ds.map(_fmt, remove_columns=raw_ds.column_names)
        ds_split = proc_ds.train_test_split(test_size=test_size, seed=seed)

        n_train  = len(ds_split['train'])
        n_test   = len(ds_split['test'])
        train_batch_idxs = range((n_train + batch_size - 1) // batch_size)
        test_batch_idxs  = range((n_test  + batch_size - 1) // batch_size)

        # 初始化父類別
        super().__init__(
            dataset_name, source_dataset_name, dataset_version,
            has_valid, split_map,
            batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None
        )

        # 保留處理完的 datasets，給 load_from_source 直接回傳
        self._datasets = ds_split

    def load_from_source(self):
        """
        若已先用 data_transfer.py 產生 json，可改用 self.load_from_json()
        這裡直接回傳記憶體中的 DatasetDict，之後 main 會呼叫 to_json() 寫檔。
        """
        return self._datasets
    
    def _post_process(self, datasets):
        """
        這裡不需要額外處理，因為在 load_from_source() 時已經格式化過了。
        """
        if 'id' in datasets['train'].column_names:
            datasets = datasets.remove_columns(['id'])
        return datasets
    
    def _first_to_str(g):
        """把 list[dict] 或 list[str] 的第一個元素轉成純文字"""
        if isinstance(g, str):
            return g
        if isinstance(g, dict):
            return g.get("content") or g.get("text") or json.dumps(g)
        return json.dumps(g)

    def _parse_llm_output(self, output: str):
        """
        回傳 (rationale, label)，能處理：
            list[str]               → ["完整思路 …"]
            list[list[str]]         → [["思路1 …", "思路2 …", …]]
            dict / str              → 舊格式保底
        """

        # ---------- 1. 還原最外層 ---------- #
        try:
            data = json.loads(output)
        except Exception:
            data = output                                  # 已是純字串

        # ---------- 2. 抽出欲解析的文字 ---------- #
        # 若 data 是 list[list[str]]：取 data[0][0]（或 data[0][-1] 亦可）
        # 若 data 是 list[str]     ：取 data[0]
        # 其它則轉成字串
        if isinstance(data, list):
            first = data[0] if data else ""
            if isinstance(first, list):
                raw = first[0] if first else ""
            elif isinstance(first, str):
                raw = first
            else:                          # list[dict] 等極端情況
                raw = json.dumps(first)
        elif isinstance(data, str):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("content") or data.get("text") or json.dumps(data)
        else:
            raw = str(data)

        # ---------- 3. 擷取 label ---------- #
        def _match(regex):
            return re.search(regex, raw, re.S | re.I)
        
        ans = list(re.finditer(r'Answer:\s*([^\n]+)', raw))
        if ans:
            label     = ans[-1].group(1).strip()
            rationale = raw[:ans[-1].start()].rstrip()
            print(f"Warning: 使用 'Answer:' 標記解析，可能不符合預期：{label}")
            return rationale, label

        m = _match(r'Final Answer.*?\\boxed\{([^}]*)\}')
        if m:
            label     = m.group(1).strip()
            rationale = raw[:m.start()].rstrip()
            return rationale, label

        m_all = list(re.finditer(r'\\boxed\{([^}]*)\}', raw))
        if m_all:
            label     = m_all[-1].group(1).strip()
            rationale = raw[:m_all[-1].start()].rstrip()
            return rationale, label

        
        raise ValueError(
            f"無法解析輸出：{output}\n"
            f"請檢查輸出格式是否符合預期，或更新解析邏輯。"
        )
        # ---------- 4. 全部失敗 ---------- #
        return raw.strip(), ""

    # GPT-NeoX 版與 LLM 版解析邏輯相同，可共用
    def _parse_gpt_output(self, output: str):
        return self._parse_llm_output(output)

class HendrycksMathDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'hendrycks_math'
        source_dataset_name = 'algebra'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(15)
        test_batch_idxs = range(10)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.source_dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.source_dataset_name}_test.json',
        }

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets

    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/algebra_train.json') as f:
            original_train_dataset = json.load(f)
        with open(f'{self.data_root}/{self.dataset_name}/algebra_test.json') as f:
            original_test_dataset = json.load(f)
        train_dataset = [
            {
                'input': data['input'],
                'label': str(data['label']),
            } for data in original_train_dataset
        ]

        test_dataset = [
            {
                'input': data['input'],
                'label': str(data['label']),
            } for data in original_test_dataset
        ]

        train_dataset = Dataset.from_list(train_dataset)
        test_dataset = Dataset.from_list(test_dataset)

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets
        

    def _post_process(self, datasets):
        datasets = datasets.remove_columns(['process'])
        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()

        label = label.strip()

        if label.endswith('.'):
            label = label[:-1]

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()

        label = label.strip()

        return rationale, label

class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/SVAMP.json') as f:
            original_dataset = json.load(f)

        dataset = list()
        for data in original_dataset:
            input = f'{data["Body"]}\n{data["Question"]}'
            equation = data["Equation"]

            dataset.append({
                'input': input,
                'label': equation,
            })

        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        train_idxs = idxs[:800]
        test_idxs = idxs[800:]

        train_dataset = Dataset.from_list(np.array(dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets
        

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label


class ASDivDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'asdiv'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 1000
        train_batch_idxs = range(3)
        test_batch_idxs = range(1)

        super().__init__(dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        raise NotImplementedError
        

    def _post_process(self, datasets):

        def prepare_input(example):
            example['input'] = example['Body'] + '\n' + example['Question']
            answer = example['Answer'].split(' ')[0]
            example['label'] = answer

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['Body', 'Question', 'Formula', 'Answer'])

        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class ESNLIDatasetLoader(DatasetLoader):
    def __init__(self, subset='full'):
        dataset_name = 'esnli'
        source_dataset_name = 'esnli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 5500
        if subset == 'full':
            train_batch_idxs = range(100)
        elif subset == 'small':
            train_batch_idxs = range(10)
        else:
            raise ValueError
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['explanation_1', 'explanation_2', 'explanation_3'])

        return datasets


    def _parse_llm_output(self, output):
        rationale = output.split("Answer:")[0].rstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label

    
    def _parse_gpt_output(self, output):
        rationale = output.split("Answer:")[0].rstrip().lstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label


class ANLIDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs):

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):
        
        def label_idx2text(example):            
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'
            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(['uid', 'reason'])

        return datasets


    def _parse_llm_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().split("So the answer is")
        except:
            rationale = ''
            label = ''
        
        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label


    def _parse_gpt_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("So the answer is")
        except:
            try:
                rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("The answer is")
            except:
                rationale = ''
                label = ''

        
        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label


class ANLI1DatasetLoader(ANLIDatasetLoader):
    def __init__(self):
        dataset_name = 'anli1'
        source_dataset_name = 'anli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train_r1',
            'valid': 'dev_r1',
            'test': 'test_r1',
        }
        batch_size = 5000
        train_batch_idxs = range(4)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
    elif args.dataset == 'OpenR1-Math-220k':            # ⬅️ 新增
        dataset_loader = OpenR1Math220kDatasetLoader()

    datasets = dataset_loader.load_from_source()
    dataset_loader.to_json(datasets)
