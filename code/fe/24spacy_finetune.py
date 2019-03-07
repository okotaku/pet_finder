import spacy
from spacy.gold import GoldParse
from utils import *


def spacy_d2v(train_text):
    nlp = spacy.load('en_core_web_md')
    new_tags = [None] * len(train_text)
    new_tags[0] = 'VBP'
    gold = GoldParse(train_text, tags=new_tags)
    nlp.update(train_text, gold, update_shared=True)

    result = np.array([nlp(text).vector for text in train["Description"].values])

    d2v_cols = ["spacy_d2v_md_finetune{}".format(i) for i in range(1, result.shape[1] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


if __name__ == '__main__':
    result = spacy_d2v(train["Description"])
    result.to_feather("../feature/spacy_d2v_md_finetune.feather")