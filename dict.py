import json
import os

from nltk.tokenize import TweetTokenizer

UNK_TOKEN = '__UNK__'
START_TOKEN = '__START__'
END_TOKEN = '__END__'
PAD_TOKEN = '__PAD__'
SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


def split_tokenize(text):
    """Splits tokens based on whitespace after adding whitespace around
    punctuation.
    """
    return (text.lower().replace('.', ' . ').replace('. . .', '...')
            .replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ')
            .replace('!', ' ! ').replace('?', ' ? ')
            .split())


class Dictionary:

    def __init__(self, file=None, min_freq=0, split=False):
        self.i2tok = list()
        self.tok2i = dict()
        self.tok2cnt = dict()
        self.split = split

        for tok in SPECIALS:
            self.tok2i[tok] = len(self.tok2i)
            self.i2tok.append(tok)
            self.tok2cnt[tok] = 100000000

        if file is not None:
            with open(file) as f:
                for line in f:
                    try:
                        tok, cnt = line.split('\t')
                    except:
                        tok, cnt = line.split(' ')
                    if int(cnt) >= min_freq:
                        self.tok2i[tok] = len(self.i2tok)
                        self.tok2cnt[tok] = int(cnt)
                        self.i2tok.append(tok)

        self.tokenizer = TweetTokenizer()

    def __len__(self):
        return len(self.i2tok)

    def __getitem__(self, tok):
        return self.tok2i.get(tok, self.tok2i[UNK_TOKEN])

    def encode(self, msg, include_end=False):
        if self.split:
            ret = [self[tok] for tok in split_tokenize(msg)]
        else:
            ret = [self[tok] for tok in self.tokenizer.tokenize(msg)]
        return ret + [self[END_TOKEN]] if include_end else ret

    def decode(self, toks):
        res = []
        for tok in toks:
            tok = self.i2tok[tok]
            if tok != END_TOKEN:
                res.append(tok)
            else:
                break
        return ' '.join(res)

    def add(self, msg):
        # for tok in self.tokenizer.tokenize(msg):
        for tok in split_tokenize(msg):
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
    # data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')
    data_dir = './data'
    train_set = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
    valid_set = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
    test_set = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

    # dictionary = Dictionary('./data/dict.txt', 3)

    dictionary = Dictionary()
    for set in [train_set, valid_set, test_set]:
        for config in set:
            for msg in config['dialog']:
                if msg['id'] == 'Tourist':
                    if msg['text'] not in ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']:
                        if len(msg['text'].split(' ')) > 2:
                            dictionary.add(msg['text'])

    dictionary.save('./data/tourist_lower_dict_gt2.txt')
