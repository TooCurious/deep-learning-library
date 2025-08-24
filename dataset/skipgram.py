from torch.utils.data import Dataset
import numpy as np

class SkipGramDataset(Dataset):

    def __init__(self, config, text):

        self.config = config
        self.text = text.split()

        self._vocab()

        self.center_word, self.context_word = self._skip_gram()



    def _vocab(self):
        vocab = set(self.text)
        dict_uniq_word = {item: index for index, item in enumerate(vocab)}
        self._vocabulary = dict_uniq_word


    def _word_to_index(self, word):

        return self._vocabulary.get(word)



    def _skip_gram(self):

        center_word = []
        context_word = []

        for i in range(self.config.context_size, len(self.text) - self.config.context_size):
            array = [self._word_to_index(self.text[j])
                     for j in np.arange(i - self.config.context_size, i + self.config.context_size + 1)
                     if j != i]
            center_word.extend([self._word_to_index(self.text[i])] * self.config.context_size)
            context_word.extend(array)

        return center_word, context_word


    def vocab_len(self):
        return len(self._vocabulary)


    def __len__(self):
        return len(self.center_word)


    def __getitem__(self, idx: int) -> dict:
        return {'center_word': self.center_word[idx], 'context_word': self.context_word[idx]}




