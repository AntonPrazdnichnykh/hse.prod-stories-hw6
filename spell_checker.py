import hunspell
from n_grams import BiGramModel
from textdistance import levenshtein, jaro_winkler
from typing import Optional
from collections.abc import Callable
import re
from nltk import pos_tag


class SpellChecker:
    def __init__(
            self,
            url1: str = 'https://www.norvig.com/ngrams/count_1w.txt',
            url2: str = 'https://www.norvig.com/ngrams/count_2w.txt',
            range_fn: Optional[Callable] = None,
            mode: str = 'min',
    ):
        self.spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.bigram_model = BiGramModel(url1, url2)
        self.range_fn = range_fn if range_fn is not None else mean
        assert mode in ['min', 'max']
        self.mode = mode


    def get_fearures(self, triplet, candidates):
        prev, wrong, foll = triplet
        features = [
            (
                levenshtein(wrong, cand),
                1 - jaro_winkler(wrong, cand),
                self.bigram_model.neg_logprob_seq_bi([cand, foll], sos=prev)
            ) for cand in candidates
        ]
        return features

    def _range(self, features):
        if self.mode == 'min':
            return min(range(len(features)), key=lambda idx: self.range_fn(features[idx]))
        return max(range(len(features)), key= lambda idx: self.range_fn(features[idx]))

    def add_to_dict(self, words):
        for w in words:
            self.spellchecker.add(w)

    def correct_words(self, text):
        words_pos = [(m.start(), m.end()) for m in re.finditer(r'\w+', text)]
        words_tagged = pos_tag(re.findall(r'\w+', text))
        corrected = []
        words_tagged = [('<S>', 'SOS')] + words_tagged + [('<E>', 'EOS')]
        for i in range(1, len(words_tagged) - 1):
            word, tag = words_tagged[i]
            if not word.isalpha() or tag == 'NNP' or self.spellchecker.spell(word):
                corrected.append(word)
            else:
                prev = corrected[-1] if corrected else words_tagged[-1][0]
                foll, _ = words_tagged[i + 1]
                suggestions = self.spellchecker.suggest(word)
                best_idx = self._range(self.get_fearures((prev, word, foll), suggestions))
                corrected.append(suggestions[best_idx] if suggestions else word)
        text_corrected = ''
        last_idx = 0
        for (start, end), word in zip(words_pos, corrected):
            text_corrected += (text[last_idx: start] + word)
            last_idx = end
        return text_corrected


def mean(x):
    return sum(x) / len(x)


if __name__ == "__main__":
    spellchecker = SpellChecker()
    text = "Zis tekst contains som erors."
    print(spellchecker.correct_words(text))
