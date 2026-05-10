import json
import numpy as np
import re
import string
import spacy
import nltk
import os
from nltk.tokenize import sent_tokenize
from fact_score.openai_lm import OpenAIModel
from rank_bm25 import BM25Okapi

nltk.download("punkt_tab", quiet=True)


class AtomicFactGenerator(object):
    def __init__(self, model_name, cache_dir_prefix='.'):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.lm = OpenAIModel(model_name=model_name)
        self.demo_path = 'fact_score/demos/demos_complex.json'

        with open(self.demo_path) as f:
            self.demos = json.load(f)

        self.demo_keys = list(self.demos.keys())
        tokenized_corpus = [doc.split(" ") for doc in self.demo_keys]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.sent_cache_fp = os.path.join(cache_dir_prefix, 'sent2facts_cache.json')
        if os.path.exists(self.sent_cache_fp):
            with open(self.sent_cache_fp) as f:
                self.sent_cache = json.load(f)
            assert isinstance(self.sent_cache, dict)
        else:
            self.sent_cache = {}

    def extract_facts(self, generation, cost_estimate=None):
        assert isinstance(generation, str)
        generation = re.sub(r' (?=[A-Z][a-z]+:)', '. ', generation)
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]

        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))
            paragraph = re.sub(r'\.\.+', '<ellipsis>. ', paragraph)
            curr_sentences = [r.strip() + '.' for s in paragraph.split('. ') for r in sent_tokenize(s)]
            sentences = [s.replace('<ellipsis>', '...') for s in sentences]
            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(sentences)

        if cost_estimate:
            return atoms_or_estimate

        atoms = atoms_or_estimate
        atomic_facts_pairs = []
        for sent in sentences:
            if any(sent.startswith(x) for x in ['Sure', 'Here are', 'Please', 'I hope']):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)
        return atomic_facts_pairs

    def get_init_atomic_facts_from_sentence(self, sentences):
        k = 1
        n = 7

        atoms = {}
        for sent_idx, sentence in enumerate(sentences):
            if sentence in atoms or sentence in self.sent_cache:
                if sentence in self.sent_cache:
                    atoms[sentence] = self.sent_cache[sentence]
                continue

            top_matchings = best_demos(sentence, self.bm25, self.demo_keys, k)
            prompt = ""

            for demo_idx in range(n):
                key = self.demo_keys[demo_idx]
                prompt += f"Please breakdown the following sentence into independent facts: {key}\n"
                for fact in self.demos[key]:
                    prompt += f"- {fact}\n"
                prompt += "\n"

            for match in top_matchings:
                prompt += f"Please breakdown the following sentence into independent facts: {match}\n"
                for fact in self.demos[match]:
                    prompt += f"- {fact}\n"
                prompt += "\n"

            if sentence.split()[0] in ('The', 'A') and len(sentence.split()) == 2:
                atoms[sentence] = ['<MALFORMED SENTENCE>']
            else:
                if is_first_or_second_person(sentence):
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is first person')
                    continue
                if '?' in sentence:
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is question')
                    continue
                if ':' in sentence:
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is script-like line')
                    continue
                prompt += f"Please breakdown the following sentence into independent facts: {sentence}\n"
                prompt += "\nOnly generate facts starting from '-' without other prefixes: "
                output = self.lm.generate(prompt, max_output_tokens=64)
                if not output.startswith('-'):
                    if sent_idx == len(sentences) - 1:
                        atoms[sentence] = []
                    else:
                        atoms[sentence] = ['<MALFORMED SENTENCE>']
                else:
                    maybe_facts = text_to_sentences(output)
                    atoms[sentence] = maybe_facts

            self.sent_cache[sentence] = atoms[sentence]

        with open(self.sent_cache_fp, 'w') as f:
            json.dump(self.sent_cache, f)

        for key, value in self.demos.items():
            if key not in atoms:
                atoms[key] = value

        return atoms


def is_first_or_second_person(s):
    s = s.replace('"', '')
    s = s.replace("'", " '")
    found = False
    for w in ('And', 'But', 'So', 'Well', 'Because'):
        if s.startswith(w):
            body = s[len(w):]
            found = True
    if not found:
        body = s[1:]
    if 'we' not in s.lower().split() and 'I' not in s.split() and 'you' not in s.split():
        return False
    return body.lower() == body.replace('I', 'i')


def best_demos(query, bm25, demos_sents, k):
    tokenized_query = query.split(" ")
    return bm25.get_top_n(tokenized_query, demos_sents, k)


def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip() and sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if sentences and sentences[-1] and sentences[-1][-1] != '.':
        sentences[-1] = sentences[-1] + '.'
    return sentences


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.UNICODE)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


MONTHS = [m.lower() for m in [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]]


def is_num(text):
    try:
        int(text)
        return True
    except Exception:
        return False


def is_date(text):
    text = normalize_answer(text)
    return all(is_num(token) or token in MONTHS for token in text.split())


def extract_numeric_values(text):
    return set(re.findall(r'\b\d+\b', text))


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for part in text.split("-"):
                entities.add(part.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities


def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    new_atomic_facts = []
    for sent, facts in atomic_facts:
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        new_facts = []
        for fact_idx, fact in enumerate(facts):
            if any(fact.endswith(v) for v in verbs) and not any(fact.endswith(v) for v in permitted_verbs):
                if any(fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != fact_idx):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= {e for e in sent_entities if e in entities}
            new_entities = sent_entities - entities
            if new_entities:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = next((ent for ent in entities if ent.startswith(new_ent)), None)
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact not in new_facts:
                new_facts.append(fact)
        try:
            assert entities == covered_entities
        except Exception:
            new_facts = facts
        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks
