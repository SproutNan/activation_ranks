from dataclasses import dataclass
from datasets import load_dataset
import random

@dataclass
class TextBatch:
    texts: list[str]

@dataclass
class TextBatchList:
    batches: list[TextBatch] = None

    def __init__(self):
        self.batches = []

class RawTextCorpus:
    def __init__(self, name: str, texts: list[str]):
        self.texts: list[str] = texts
        self.name: str = name

    def __len__(self):
        return len(self.texts)

    def GetRawBatches(self, num_examples_each, batch_size=10) -> tuple[TextBatchList, TextBatchList]:
        if num_examples_each * 2 > len(self.texts):
            raise ValueError(f'num_examples_each * 2 should be less than {len(self.texts)}')

        samples1, samples2 = self.texts[:num_examples_each], self.texts[num_examples_each:num_examples_each*2]
        assert len(samples1) % batch_size == 0 and len(samples2) % batch_size == 0, 'num_examples_each should be divisible by batch_size'
        
        batches_1, batches_2 = TextBatchList(), TextBatchList()
        for i in range(0, len(samples1), batch_size):
            batches_1.batches.append(TextBatch(samples1[i:i+batch_size]))
            batches_2.batches.append(TextBatch(samples2[i:i+batch_size]))
        
        return batches_1, batches_2

####################################################################################################

def TextCorpusDispatcher(corpus_name: str, num_each: int, batch_size: int=10) -> tuple[TextBatchList, TextBatchList]:
    dataset = load_dataset(corpus_name, split='train')
    texts = [item['text'] for item in dataset]

    if num_each * 2 > len(texts):
        raise ValueError(f'num_each * 2 should be less than {len(texts)}')
    
    random.shuffle(texts)

    # FIXME: corpus.raw builder maybe should be a function, since it may not be the same for all datasets
    text_corpus = RawTextCorpus(corpus_name, texts[:num_each*2])

    return text_corpus.GetRawBatches(
        num_examples_each=num_each,
        batch_size=batch_size
    )
