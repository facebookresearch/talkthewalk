import json
import os

from nltk.tokenize import TweetTokenizer

UNK_TOKEN = '__UNK__'
START_TOKEN = '__START__'
END_TOKEN = '__END__'
PAD_TOKEN = '__PAD__'
SPECIALS = [UNK_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN]

class Dictionary:

    def __init__(self, file=None, min_freq=0):
        self.i2tok = list()
        self.tok2i = dict()
        self.tok2cnt = dict()

        for tok in ['__UNK__', '__START__', '__END__', '__PAD__']:
            self.tok2i[tok] = len(self.tok2i)
            self.i2tok.append(tok)
            self.tok2cnt[tok] = 100000000

        if file is not None:
            with open(file) as f:
                for line in f:
                    tok, cnt = line.split('\t')
                    if int(cnt) >= min_freq:
                        self.tok2i[tok] = len(self.i2tok)
                        self.tok2cnt[tok] = int(cnt)
                        self.i2tok.append(tok)

        self.tokenizer = TweetTokenizer()

    def __len__(self):
        return len(self.i2tok)

    def __getitem__(self, tok):
        return self.tok2i.get(tok, self.tok2i['__UNK__'])

    def encode(self, msg, include_end=False):
        ret = [self[tok] for tok in self.tokenizer.tokenize(msg)]
        return ret + [self[END_TOKEN]] if include_end else ret

    def decode(self, toks):
        res = [self.i2tok[i] for i in toks if self.i2tok[i] not in SPECIALS]
        return ' '.join(res)

    def add(self, msg):
        for tok in self.tokenizer.tokenize(msg):
            if tok not in self.tok2i:
                self.tok2cnt[tok] = 0
                self.tok2i[tok] = len(self.i2tok)
                self.i2tok.append(tok)
            self.tok2cnt[tok] += 1

    def save(self, file):
        toklist = [(tok, cnt) for tok, cnt in self.tok2cnt.items()]
        sorted_list = sorted(toklist, key=lambda x: x[1], reverse=True)

        with open(file, 'w') as f:
            for tok in sorted_list:
                f.write(tok[0] + '\t' + str(tok[1]) + '\n')


if __name__ == '__main__':
    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_set = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
    valid_set = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
    test_set = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

    # dictionary = Dictionary('./data/dict.txt', 3)

    dictionary = Dictionary()
    for set in [train_set, valid_set, test_set]:
        for config in set:
            for msg in config['dialog']:
                if msg['text'] not in ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']:
                    dictionary.add(msg['text'])

    dictionary.save('./data/dict.txt')
