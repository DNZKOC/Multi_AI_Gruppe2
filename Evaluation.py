import json
import numpy as np
import scipy
import pandas as pd
from googletrans import Translator
import random


def get_eng_data(train_path, test_path, amount=-1):

    print("Start Reading en Data")

    paths = [train_path, test_path]
    results = []

    # Json Datei wird eingelesen und in das richitge Format gebracht.
    for path in paths:
        with open(path, 'r') as test:
            data = test.read()

        values = []
        decoder = json.JSONDecoder()
        while data:
            value, new_start = decoder.raw_decode(data)
            data = data[new_start:].strip()
            values.append(value)

        # Daten randomisiert wählen.
        random.shuffle(values)

        # Die JSON-Datei wird umstrukturiert, da man so einfacher nach Tweets über die id filtern kann.
        result = {}
        for item in values:
            item_dict = {"text": item["text"], "date": item["date"], "label": item["label"],
                         "label_name": item["label_name"]}
            result[item["id"]] = item_dict

        # Angegebene Menge an Trainingsdaten benutzen.
        if amount != -1 and path == train_path:

            train_result = dict(list(result.items())[:amount])
            results.append(train_result)

        # Für jedes Label 100 Testdaten benutzen.
        elif amount != -1 and path == test_path:
            test_result = {}
            counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

            for id, values in result.items():
                if values["label"] == 0 and counts["0"] < 100:
                    test_result[id] = values
                    counts["0"] += 1
                elif values["label"] == 1 and counts["1"] < 100:
                    test_result[id] = values
                    counts["1"] += 1
                elif values["label"] == 2 and counts["2"] < 100:
                    test_result[id] = values
                    counts["2"] += 1
                elif values["label"] == 3 and counts["3"] < 100:
                    test_result[id] = values
                    counts["3"] += 1
                elif values["label"] == 4 and counts["4"] < 100:
                    test_result[id] = values
                    counts["4"] += 1
                elif values["label"] == 5 and counts["5"] < 100:
                    test_result[id] = values
                    counts["5"] += 1
            results.append(test_result)

    return results[0], results[1]


def get_eng_embedding(train_data, test_data, amount, train_path, test_path):
    """
    Liest Embeddings ein. Danach werden alle rausgeworfen die nicht in test und trainingsdaten sind.
    """
    train_embedding = np.load(train_path, allow_pickle=True).item()
    test_embedding = np.load(test_path, allow_pickle=True).item()

    # Für alle Tweets die benutzt werden, werden die Embeddings zu den Daten ins dictionary gepackt.
    for tweet_id, vector in train_embedding.items():
        try:
            train_data[tweet_id]["embedding"] = vector
        except KeyError:
            pass

    for tweet_id, vector in test_embedding.items():
        try:
            test_data[tweet_id]["embedding"] = vector
        except KeyError:
            pass

    print("Finished Reading en Data")

    return train_data, test_data


def get_annotated_data(path, lang):
    """
    Liest annotierte Daten für fr, tr und de ein.
    """
    print("Start Reading", lang, "Data")

    with open(path) as f:
        data = json.load(f)

    # Filtern der Daten die nicht annotiert wurden.
    new_data = []
    for elem in data:
        if elem["label"] != -1:
            new_data.append(elem)
    data = new_data

    # Umstrukturieren der Daten damit man besser nach tweets mit bestimmter id suchen kann.
    result = {}
    for obj in data:
        key = obj['id']
        value = {k: v for k, v in obj.items() if k != 'id'}
        result[key] = value

    return result


def read_test_data(path, amount, lang):
    """
    Liest nicht annotierte Daten für fr, tr und de ein und übersetzt diese.
    """
    with open(path) as f:
        data = json.load(f)

    data = dict(random.sample(list(data.items()), amount))
    translator = Translator()
    for id, values in data.items():
        text_en = translator.translate(values["text"], src=lang, dest='en').text
        values["text_tl"] = text_en

    return data


def combine_embedding_with_data(data, test_data, emb_path, lang):
    """
    Packt embedding zu tweets zu den anderen values (text, label, etc.) ins dict.
    """
    embeddings = np.load(emb_path, allow_pickle=True).item()

    for id in data.keys():
        for tweet_id, vector in embeddings.items():
            if id == tweet_id:
                data[id]["embedding"] = vector
    for id in test_data.keys():
        for tweet_id, vector in embeddings.items():
            if id == tweet_id:
                test_data[id]["embedding"] = vector

    print("Finished Reading", lang, "Data.")

    return data, test_data


def calculate_distance(metric, vector, compare_data):
    """
    Berechnet die Distanz der als Argument übergebenen Metrik eines Vektos zu einer Menge von Vektoren.
    """
    distance_list = []
    for id, value in compare_data.items():
        if metric == "cos":
            dist = scipy.spatial.distance.cosine(vector, value["embedding"])
        elif metric == "eucl":
            dist = scipy.spatial.distance.euclidean(vector, value["embedding"])
        elif metric == "braycurtis":
            dist = scipy.spatial.distance.braycurtis(vector, value["embedding"])
        elif metric == "canberra":
            dist = scipy.spatial.distance.canberra(vector, value["embedding"])
        elif metric == "chebyshev":
            dist = scipy.spatial.distance.chebyshev(vector, value["embedding"])
        elif metric == "city_block":
            dist = scipy.spatial.distance.cityblock(vector, value["embedding"])
        distance_list.append([id, dist])

    # Liste wird nach Distanz sortiert und die 50 nähesten Tweets werden zurückgegeben.
    sorted_list = sorted(distance_list, key=lambda x: x[1])
    sorted_list = sorted_list[:50]

    return sorted_list


def get_label_average_distance(metric, vector, compare_data, label_lst):
    """
    Weist dem übergebenem Vektor ein label zu. Dabei wird die average Distanz des Vekors zu allen Vektoren eines
    bestimmten Labels berechnet. Dabei werden nur Labels berücksichtigt die mindestens 9 mal in der Menge der
    50 nähesten Vekoren vertreten sind.
    """

    distance_list = calculate_distance(metric, vector, compare_data)

    counts = {k: 0 for k in label_lst}
    distance_sum = {d: 0 for d in label_lst}

    # Die Anzahl der jeweiligen Labels wird gezählt und die Distanz dieser aufsummiert.
    for idx, dist in distance_list:
        counts[compare_data[idx]["label_name"]] += 1
        distance_sum[compare_data[idx]["label_name"]] += dist

    # Filtern aller Labels die weniger als 9 mal vorkommen.
    for label, count in counts.items():
        if count < 9:
            del distance_sum[label]

    best_label = ""
    min_dist = 10000

    # Label mit niedrigster average distance bestimmen.
    for label, dist in distance_sum.items():
        average_dist = dist / counts[label]
        if average_dist < min_dist:
            min_dist = average_dist
            best_label = label

    return best_label


def get_label_reciprocal(metric, vector, compare_data, label_lst):
    """
    Weist dem übergebenem Vektor ein label zu. Dabei werden die  Distanzen des Vektors zu den 50 nähesten Vektoren
    berechnet. Für jedes Label wird nun der Kehrwert der Distanzen aller Vektoren mit dem bestimmten Label aufsummiert.
    Das Label bei dem diese Summe am größten ist wird dem Vektor zugewiesen.
    """

    distance_list = calculate_distance(metric, vector, compare_data)
    distance_reciprocal_sum = {r: 0 for r in label_lst}

    # Kehrwerte der Distanzen aufsummieren.
    for idx, dist in distance_list:
        distance_reciprocal_sum[compare_data[idx]["label_name"]] += 1/dist

    # Label mit maximaler Summe zurückgeben.
    best_label = (max(distance_reciprocal_sum, key=distance_reciprocal_sum.get))

    return best_label


def get_label_majority_vote(metric, vector, compare_data, label_lst):
    """
    Weist dem übergebenem Vektor ein label zu. Dabei werden die  Distanzen des Vektors zu den 50 nähesten Vektoren
    berechnet. Das Label was am häufgisten in der Menge dieser 50 Vektoren auftritt wird dem Vektor zugewiesen.
    """

    distance_list = calculate_distance(metric, vector, compare_data)
    counts = {k: 0 for k in label_lst}

    # Anzahl der Tweets mit bestimmten Label wird gezählt.
    for idx, dist in distance_list:
        counts[compare_data[idx]["label_name"]] += 1

    # Häufigstes Label wird zurückgegeben.
    majority_label = (max(counts, key=counts.get))

    return majority_label


def eval_labels(data, test_data, lang, amount, path_dest=""):
    """
    Berechnet für alle Daten in test_data das zugehörige Label. Für die englischen Daten werden die Ergebnisse in der
    Konsole ausgegeben, für die anderen Sprachen wird eine Excel Datei erstellt die dann manuell ausgewertet werden
    muss.
    """
    print("")
    print("Start Evaluation of", lang, "data." )
    print("")

    counter = 0
    metrics = ["cos", "eucl", "braycurtis", "canberra", "chebyshev", "city_block"]

    if lang == "en":
        poss_labels = ["arts_&_culture", "business_&_entrepreneurs", "pop_culture", "daily_life", "sports_&_gaming",
                       "science_&_technology"]
        metric_results = {metric: 0 for metric in metrics}
        metric_results["overall"] = 0
    else:
        poss_labels = ["arts_and_culture", "business_and_entrepreneurs", "pop_culture", "daily_life",
                       "sports_and_gaming", "science_and_technology", "politics"]

        df = pd.DataFrame(columns=["tweet", "tweet übers.", "cosinus", "cosinus eval", "euclid", "euclid eval",
                                   "braycurtis", "braycurtis eval", "canberra", "canberra eval", "chebyshev",
                                   "chebyshev eval", "cityblock", "cityblock eval", "Majority", "Majority Eval"])

    # Iterien durch die Menge aller zu evaluierenden Tweets.
    for tweet_id, value in test_data.items():
        lable_amount = {label_: 0 for label_ in poss_labels}
        labels = {}

        # Berechnen des Labels für jede Metrik.
        for metric in metrics:
            labels[metric] = get_label_majority_vote(metric, value["embedding"], data, poss_labels)
            lable_amount[labels[metric]] += 1

        # Majority Vote über alle Labels der einzelnen Metriken.
        overall_label = max(lable_amount, key=lable_amount.get)

        if lang == "en":
            tweet_label = test_data[tweet_id]["label_name"]
            if overall_label == tweet_label:
                metric_results["overall"] += 1
            for dist, label in labels.items():
                if label == tweet_label:
                    metric_results[dist] += 1
        else:
            df = df.append({"tweet": value["text"], "tweet übers.": value["text_tl"], "cosinus": labels["cos"],
                            "euclid": labels["eucl"], "braycurtis": labels["braycurtis"],
                            "canberra": labels["canberra"], "chebyshev": labels["chebyshev"],
                            "cityblock": labels["city_block"], "Majority": overall_label}, ignore_index=True)
        if counter % 50 == 0 and counter > 0:
            print(counter, "Tweets are finished analyzing.")
        counter += 1

    if lang == "en":
        print("")
        print("Correct classified Tweets out of " + str(amount))
        print("")
        for metric, result in metric_results.items():
            print(metric, ": ", result)
    else:
        df.to_excel(path_dest, index=False)


def main():
    language_lst = ["en", "de", "tr", "fr"]

    # Teilmenge an Sprachen festlegen die evaluiert werden sollen.
    languages = language_lst[2:3]

    # Anzahl an zu evaluierenden Tweets festlegen.
    tweets_to_eval = 100

    for lang in languages:
        if lang == "en":
            eng_train_path = 'tweets/annotated/train_random.single.json'
            eng_test_path = 'tweets/annotated/test_random.single.json'
            eng_emb_train_path = "embeddings/en_train_embeddings.npy"
            eng_emb_test_path = "embeddings/en_test_embeddings.npy"

            train_data, test_data = get_eng_data(eng_train_path, eng_test_path, amount=tweets_to_eval)
            train_data, test_data = get_eng_embedding(train_data, test_data, tweets_to_eval
                                                      , eng_emb_train_path, eng_emb_test_path)
            eval_labels(test_data, train_data, lang, tweets_to_eval)
        else:
            path_dest = "eval/" + lang + "_eval.xlsx"
            annotated_path = "tweets/annotated/" + lang + '_annotated.json'
            test_path = 'tweets/raw/' + lang + '_tweets.json'
            embedding_path = "embeddings/" + lang + "_embeddings.npy"

            data = get_annotated_data(annotated_path, lang)
            test_data = read_test_data(test_path, tweets_to_eval, lang)
            data, test_data = combine_embedding_with_data(data, test_data, embedding_path, lang)
            eval_labels(data, test_data, lang, tweets_to_eval, path_dest=path_dest)


if __name__ == "__main__":
    main()
