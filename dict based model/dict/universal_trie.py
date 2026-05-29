import os
import re
import time
import csv

class TrieNode:
    __slots__ = ['children', 'is_end', 'pos_tags', 'meanings']
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.pos_tags = set()
        self.meanings = []

class UniversalTrie:
    """
    Highly Specific Universal Semantic Trie.
    Merges multiple linguistic datasets to provide precise POS tags and definitions.
    Strictly rejects strings that lack verified semantic data.
    """
    
    # Expanded mapping for highly specific Part of Speech identification
    POS_MAP = {
        # Moby POS Codes
        'N': 'Noun (General)',
        'p': 'Noun (Plural)',
        'h': 'Noun Phrase',
        'V': 'Verb (General)',
        't': 'Verb (Transitive)',
        'i': 'Verb (Intransitive)',
        'A': 'Adjective',
        'v': 'Adverb',
        'C': 'Conjunction',
        'P': 'Preposition',
        '!': 'Interjection',
        'r': 'Pronoun',
        'D': 'Definite Article',
        'I': 'Indefinite Article',
        'o': 'Nominative Case',
        # CSV/Dictionary specific abbreviations
        'prep.': 'Preposition',
        'a.': 'Adjective',
        'v. t.': 'Verb (Transitive)', 
        'v. i.': 'Verb (Intransitive)', 
        'n.': 'Noun (General)', 
        'adv.': 'Adverb', 
        'conj.': 'Conjunction',
        'pron.': 'Pronoun',
        'interj.': 'Interjection',
        'p. a.': 'Participial Adjective',
        'n. pl.': 'Noun (Plural)'
    }

    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        
        # Pattern Regex
        self.url_pattern = re.compile(r'^(https?://)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$', re.IGNORECASE)
        self.date_pattern = re.compile(r'^(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|([A-Za-z]{3,9}\s\d{1,2}(st|nd|rd|th)?,?\s\d{4})$', re.IGNORECASE)
        self.money_pattern = re.compile(r'^(\$|€|£|¥|₹)?\s?\d{1,3}(,\d{3})*(\.\d{1,2})?\s?(USD|EUR|GBP|JPY|INR|bucks)?$', re.IGNORECASE)
        self.number_pattern = re.compile(r'^[+-]?\d{1,3}(,\d{3})*(\.\d+)?$')

    def insert(self, word: str, pos_info: str = None, meaning: str = None):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Word is only "Accepted" later if it has POS or Meanings
        node.is_end = True
        
        if pos_info:
            pos_info = pos_info.strip()
            if pos_info in self.POS_MAP:
                node.pos_tags.add(self.POS_MAP[pos_info])
            else:
                # Handle Moby string of characters (e.g. 'NVt')
                for char in pos_info:
                    if char in self.POS_MAP:
                        node.pos_tags.add(self.POS_MAP[char])
        
        if meaning:
            meaning = meaning.strip()
            if meaning and meaning not in node.meanings:
                node.meanings.append(meaning)

    def search_word(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                return False, [], []
            node = node.children[char]
        
        # Valid ONLY if it has a POS tag or a definition
        if node.is_end and (node.pos_tags or node.meanings):
            return True, list(node.pos_tags), node.meanings
        return False, [], []

    def analyze(self, token: str) -> dict:
        clean_token = token.strip()
        lower_token = clean_token.lower()
        
        # 1. Dictionary Check (Strict)
        found, tags, meanings = self.search_word(lower_token)
        if found:
            return {
                "accepted": True,
                "types": sorted(tags) if tags else ["General Vocabulary"],
                "meanings": meanings if meanings else ["Definition verified via Part-of-Speech membership."]
            }
            
        # 2. Pattern Check
        if self.url_pattern.match(clean_token):
            return {"accepted": True, "types": ["URL / Web Link"], "meanings": ["Universal Resource Locator."]}
        if self.date_pattern.match(clean_token):
            return {"accepted": True, "types": ["Date Format"], "meanings": ["A specific calendar date reference."]}
        if self.money_pattern.match(clean_token):
            return {"accepted": True, "types": ["Currency / Money"], "meanings": ["A monetary amount or financial value."]}
        if self.number_pattern.match(clean_token):
            return {"accepted": True, "types": ["Numerical Value"], "meanings": ["A mathematical or quantitative number."]}
            
        return {"accepted": False, "types": ["UNKNOWN"], "meanings": []}

    def load_dictionaries(self, base_dir: str):
        print("Building High-Precision Semantic Trie...")
        start_time = time.time()
        
        # 1. Load Meanings CSV first (Highest Quality)
        dict_csv_path = os.path.join(base_dir, 'english_dictionary.csv')
        if os.path.exists(dict_csv_path):
            print("  -> Syncing Lexical Definitions...")
            with open(dict_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4:
                        self.insert(row[1].strip().lower(), row[2].strip(), row[3].strip())

        # 2. Load Moby POS dataset (Adds grammatical specificity)
        pos_path = os.path.join(base_dir, 'mobypos.txt')
        if os.path.exists(pos_path):
            print("  -> Syncing Grammatical Specificity (Moby POS)...")
            with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '\\' in line:
                        parts = line.strip().split('\\')
                        if len(parts) == 2:
                            self.insert(parts[0].lower(), parts[1])

        # 3. Load web words (Only to mark existence, will not be 'Accepted' without POS/Meaning)
        for fname in ['words_alpha.txt', 'web_words_1M.txt']:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                print(f"  -> Indexing Background Vocabulary ({fname})...")
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            # We insert but don't add POS or Meaning here
                            node = self.root
                            for char in word:
                                if char not in node.children: node.children[char] = TrieNode()
                                node = node.children[char]
                            node.is_end = True

        self.word_count = self._count_accepted(self.root)
        end_time = time.time()
        print(f"Successfully compiled {self.word_count:,} verified entries in {end_time - start_time:.2f} seconds.")

    def _count_accepted(self, node):
        count = 1 if (node.is_end and (node.pos_tags or node.meanings)) else 0
        for child in node.children.values():
            count += self._count_accepted(child)
        return count

def build_ultimate_trie(base_dir: str = None) -> UniversalTrie:
    if base_dir is None: base_dir = os.path.dirname(__file__)
    trie = UniversalTrie()
    trie.load_dictionaries(base_dir)
    return trie

if __name__ == "__main__":
    engine = build_ultimate_trie()
    print("\n--- High-Precision Semantic Engine Ready ---")
    while True:
        try:
            user_input = input("Enter string: ").strip()
            if user_input.lower() == 'exit': break
            if not user_input: continue
            
            res = engine.analyze(user_input)
            if res["accepted"]:
                print(f"  Types: {res['types']}")
                for i, m in enumerate(res['meanings'], 1):
                    display_m = (m[:200] + '...') if len(m) > 200 else m
                    print(f"  Meaning {i}: {display_m}")
            else:
                print("  Result: REJECTED (No verified linguistic data found)")
            print()
        except KeyboardInterrupt: break
