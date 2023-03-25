"""Preprocessing module for JSON tweet files."""
import json
import pathlib
import googletrans
import random
import time

MAX_RETRIES_TRANSLATION = 2


def new_tweetset(filestr):
    """
    Convert JSON file with tweets to CSV, omitting tweets with an
    amount of hashtags and urls of 30% or more of total length.
    """

    input_path = pathlib.Path(filestr)
    print('********************')

    # Parse JSON file and store in dictionary.
    tweets = _load_tweets(input_path)

    # Filter tweets.
    print('Filtering tweets…')
    filtered_tweets = _filter_tweets(json_data)
    print(f'{len(filtered_tweets)} tweets meet criteria.')

    # Pick n random tweets.
    print('Choosing random tweets…')
    random_tweets = {
        k: filtered_tweets[k] for k in random.sample(list(filtered_tweets.keys()), 5)
        }
    print(f'{len(random_tweets)} tweets randomly chosen.')

    # Translate tweets.
    print('Translating tweets…')
    translated_tweets = _translate_tweets(random_tweets)

    # Write dictionary to JSON file.
    print(f'Writing result…')
    _save_tweets(translated_tweets, input_path.stem + '_preprocessed.json')


def translate_tweetset(filestr):
    """Load file, translate (untranslated) tweets, save file."""
    input_path = pathlib.Path(filestr)
    json_data = _load_tweets(input_path)
    translated_tweets = _translate_tweets(json_data)
    _save_tweets(translated_tweets, input_path)


def add_to_tweetset(filestr_tweetset, filestr_source, n_tweets):
    """Add n random tweets from source to tweetset and translate."""
    # Load tweets from JSON files.
    tweetset_path = pathlib.Path(filestr_tweetset)
    tweetset = _load_tweets(tweetset_path)
    source_path = pathlib.Path(filestr_source)
    source = _load_tweets(source_path)

    # Remove intersection.
    print('Removing tweets already contained in tweetset from source…')
    clean_source = {k: v for k, v in source.items() if k not in tweetset}
    print(f'{len(clean_source)} tweets remaining in source.')

    # Filter source.
    print('Filtering tweets in source…')
    filtered_source = _filter_tweets(clean_source)
    print(f'{len(filtered_source)} tweets meet criteria.')

    # Choose random tweets.
    print('Choosing random tweets…')
    random_tweets = {k: filtered_source[k] for k\
            in random.sample(list(filtered_source.keys()), n_tweets)
        }
    print(f'{len(random_tweets)} tweets randomly chosen.')

    # Translate new tweets.
    print('Translating tweets…')
    translated_tweets = _translate_tweets(random_tweets)

    # Merge old tweetset and new tweets.
    print('Merging new tweets with tweetset…')
    new_tweetset = tweetset | translated_tweets

    # Write augmented tweetset to JSON file.
    _save_tweets(new_tweetset, tweetset_path)

    # Statistics.
    print(f'New size of tweetset: {len(new_tweetset)}')


def _load_tweets(input_path):
    """Helper function to parse JSON input file."""
    print(f'Processing file "{input_path}"…')
    with open(input_path, 'r') as f:
        json_data = json.load(f)
    print(f'{len(json_data)} tweets read!')
    return json_data


def _filter_tweets(tweets):
    """Helper function to filter tweets that contain < 30% text."""
    return {
        k: v for k, v in tweets.items()\
        if v.get('entities', None) is None
            or sum(
                [entity['end']-entity['start']\
                    for entitylist in v.get('entities', {}).values()\
                        for entity in entitylist]
                ) <= 0.7*len(v['text'])
        }

    # Alternative to nested list comprehension:
    # v['entities'].get('urls', [])\
    # + v['entities'].get('hashtags', [])
    # + v['entities'].get('mentions', [])


def _translate_tweets(tweets):
    """Helper function to translate tweet dictionary per Google translate."""
    translator = googletrans.Translator()
    translated_tweets = tweets.copy()
    translation_count = 0
    user_interrupt = False
    for k, v in translated_tweets.items():
        if v.get('translation', None):
            continue
        elif v['lang'] != 'de':
            retry_count = 0
            while retry_count <= MAX_RETRIES_TRANSLATION:
                if retry_count > 0:
                    print(f'Retry to translate in 3 seconds… (Remaining '
                        f'retries: {MAX_RETRIES_TRANSLATION - retry_count})')
                    time.sleep(3)
                try:
                    translated_tweets[k]['translation'] = translator\
                        .translate(v['text'], src=v['lang'], dest='de').text
                except Exception as e:
                    print(f'Error while translating: {e}')
                    retry_count += 1
                    continue
                except KeyboardInterrupt:
                    user_interrupt = True
                    break
                else:
                    translation_count += 1
                    break
            else:
                print('Maximum number of retries reached! Continuing '
                    'without translation. Use function translate_tweet'
                    'set on output to complete translation later.')
                break
            if user_interrupt:
                print(f'Translation interrupted by user. Skipping '
                    'remaining translations. Use function translate_tw'
                    'eetset on output to complete translation later.')
                break
        else:
            translated_tweets[k]['translation'] = None
    else:
        print('Translation finished without errors!')
    print(f'{translation_count} tweets translated.')
    return translated_tweets


def _save_tweets(tweets, output_path):
    """Helper function to save tweet dictionary as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tweets, f, ensure_ascii=False, indent=4)
    print(f'File "{output_path}" successfully written!')


def main():
    # To create (a) new tweetset(s), use:
    for filestr in ['de_tweets.json', 'fr_tweets.json', 'tr_tweets.json']:
        new_tweetset(filestr)

    # To complete an interrupted translation, use:
    # translate_tweetset('fr_tweets_preprocessed.json')

    # To add additional n tweets to an existing tweetset, use:
    # add_to_tweetset('fr_tweets_preprocessed.json', 'fr_tweets.json', 5)


if __name__ == '__main__':
    main()
