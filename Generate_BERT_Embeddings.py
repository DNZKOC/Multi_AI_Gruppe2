from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import tqdm
import json


class BertConverter:
    def __init__(self, model_name='bert-base-uncased', device_number=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_number = device_number
        self.model = AutoModel.from_pretrained(model_name).to("cpu")
        self.model.eval()

    def encode(self, sentences, ret_input=False):
        if type(sentences) == str:
            sentences = [sentences]

        encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512).to("cpu")

        output = self.model(**encoded_input)
        if ret_input:
            return output, encoded_input
        else:
            return output

    def encode_to_vec(self, sentences):
        bert_results = self.encode(sentences)
        return bert_results.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()

    def encode_word_embedding(self, sentences):
        inp = self.tokenizer(sentences, return_tensors='pt')
        out = self.model(**inp)
        out = out.last_hidden_state[0].detach().numpy()
        return out


def generate_ref_embeddings(all_tweets: dict, bert_convert: BertConverter, lang):
    id_compute = []
    text_compute = []
    text_embeddings = {}
    for tweet_id in tqdm.tqdm(all_tweets, desc="Compute sentence Embeddings of all " + lang + " Tweets"):
        id_compute.append(tweet_id)
        text_compute.append(all_tweets[tweet_id]["text"])
        if len(text_compute) >= 5:
            embeddings = bert_convert.encode_to_vec(text_compute)
            for c1, emd_i in enumerate(embeddings):
                text_embeddings[id_compute[c1]] = emd_i
            text_compute = []
            id_compute = []
        print(text_embeddings)
    if len(text_compute) > 0:
        embeddings = bert_convert.encode_to_vec(text_compute)
        for c1, emd_i in enumerate(embeddings):
            text_embeddings[id_compute[c1]] = emd_i

    return text_embeddings


def read_json(path):
    """
    Einlesen und formatieren der englischen Datensätze.
    """
    with open(path, 'r') as d:
        data = d.read()

    values = []
    decoder = json.JSONDecoder()
    while data:
        value, new_start = decoder.raw_decode(data)
        data = data[new_start:].strip()
        values.append(value)

        result = {}
        for item in values:
            item_dict = {"text": item["text"], "date": item["date"], "label": item["label"],
                         "label_name": item["label_name"]}
            result[item["id"]] = item_dict

    return result


if __name__ == '__main__':

    languages = ["de", "tr", "fr", "en"]
    for language in languages:

        if language == "en":
            test_values = read_json('tweets/annotated/' + 'test_random.single.json')
            train_values = read_json('tweets/annotated/' + 'train_random.single.json')
            test_embeddings = generate_ref_embeddings(all_tweets=test_values, bert_convert=BertConverter(),
                                                      lang=language)
            np.save("embeddings/en_test_embeddings.npy", test_embeddings)

            train_embeddings = generate_ref_embeddings(all_tweets=train_values, bert_convert=BertConverter(),
                                                       lang=language)
            np.save("embeddings/en_train_embeddings.npy", train_embeddings)

        else:
            with open("tweets/raw/" + language + '_tweets.json', 'r') as d:
                data = json.load(d)
            embeddings = generate_ref_embeddings(all_tweets=data, bert_convert=BertConverter(), lang=language)
            np.save("embeddings/" + language + '_embeddings.npy', embeddings)


