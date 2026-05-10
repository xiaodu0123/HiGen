import string
import json
import numpy as np
import os
import nltk

from fact_score.openai_lm import OpenAIModel
from nltk.corpus import names

nltk.download('names', quiet=True)

english_names = names.words('male.txt') + names.words('female.txt')


class FactScorer(object):
    def __init__(self, cache_dir_prefix='.', model_name=None):
        self.cache_dir_prefix = cache_dir_prefix
        os.makedirs(cache_dir_prefix, exist_ok=True)
        self.lm = OpenAIModel(model_name=model_name)

    def get_score(self, atomic_facts, reference, summname, topic=None, overwrite_cache=False):
        cache_dir = os.path.join(self.cache_dir_prefix, 'is_supported_factscore_caches')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{summname}.json')

        use_cache = os.path.exists(cache_path) and not overwrite_cache
        if use_cache:
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            print('No cache found at', cache_path)
            cache = {}

        print(f'\nScoring facts for {summname}\n')
        decisions = []
        bad_substrings = ['airs on', 'season finale', 'click', 'samaritans', '.com']

        for i, atom in enumerate(atomic_facts):
            atom = atom.strip()
            if topic is None:
                definition = 'Answer the question based on the given context.\n\n'
            else:
                definition = f'Answer the question about {topic} based on the given context.\n\n'
            definition += reference
            definition = ' '.join(definition.strip().split()[:3000])
            if definition[-1] not in string.punctuation:
                definition += "."
            prompt = f'{definition}\n\nInput: {atom.strip()} True or False?\nOutput:'

            maybe_names = [w for w in atom.rstrip('.').split() if w in english_names]
            if not all(n in definition for n in maybe_names):
                print(f'Names not in document: {[n for n in maybe_names if n not in definition]}')
                is_supported = False
            elif atom in atomic_facts[:i]:
                print('penalizing repeated fact')
                is_supported = False
            elif atom == '<MALFORMED SENTENCE>':
                is_supported = False
            elif any(x in atom.lower() for x in bad_substrings):
                is_supported = False
            elif atom in cache:
                is_supported = cache[atom]
            else:
                if use_cache:
                    print('atom:', atom, 'not in cache at', cache_path)
                output = self.lm.generate(prompt, max_output_tokens=1)
                is_supported = output.lower() == 'true'
                cache[atom] = bool(is_supported)

            print(atom, is_supported)
            decisions.append({"atom": atom, "is_supported": is_supported})

        score = np.mean([d["is_supported"] for d in decisions])
        with open(cache_path, 'w') as f:
            json.dump(cache, f)

        return score, decisions
