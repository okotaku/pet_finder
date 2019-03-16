from utils import *


class MetaDataParser(object):
    def __init__(self):
        # sentiment files
        train_sentiment_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_sentiment/*.json'))
        test_sentiment_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_sentiment/*.json'))
        sentiment_files = train_sentiment_files + test_sentiment_files
        self.sentiment_files = pd.DataFrame(sentiment_files, columns=['sentiment_filename'])
        self.sentiment_files['PetID'] = self.sentiment_files['sentiment_filename'].apply(
            lambda x: x.split('/')[-1].split('.')[0])

        # metadata files
        train_metadata_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_metadata/*.json'))
        test_metadata_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_metadata/*.json'))
        metadata_files = train_metadata_files + test_metadata_files
        self.metadata_files = pd.DataFrame(metadata_files, columns=['metadata_filename'])
        self.metadata_files['PetID'] = self.metadata_files['metadata_filename'].apply(
            lambda x: x.split('/')[-1].split('-')[0])

    def open_json_file(self, filename):
        with open(filename, 'r', encoding="utf-8") as f:
            metadata_file = json.load(f)
        return metadata_file

    def get_stats(self, array, name):
        stats = [np.mean, np.max, np.min, np.sum]
        result = {}
        if len(array):
            for stat in stats:
                result[name + '_' + stat.__name__] = stat(array)
        else:
            for stat in stats:
                result[name + '_' + stat.__name__] = 0
        return result

    def parse_sentiment_file(self, file):
        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = ' '.join(file_entities)

        file_sentences_text = [x['text']['content'] for x in file['sentences']]
        file_sentences_text = ' '.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

        file_sentiment.update(file_sentences_sentiment)
        file_sentiment.update({"sentiment_text": file_sentences_text})

        return pd.Series(file_sentiment)

    def parse_metadata(self, file):
        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            label_annotations = file['labelAnnotations']
            file_top_score = [x['score'] for x in label_annotations]
            file_top_desc = [x['description'] for x in label_annotations]
            dog_cat_scores = []
            dog_cat_topics = []
            is_dog_or_cat = []
            for label in label_annotations:
                if label['description'] == 'dog' or label['description'] == 'cat':
                    dog_cat_scores.append(label['score'])
                    dog_cat_topics.append(label['topicality'])
                    is_dog_or_cat.append(1)
                else:
                    is_dog_or_cat.append(0)
        else:
            file_top_score = []
            file_top_desc = []
            dog_cat_scores = []
            dog_cat_topics = []
            is_dog_or_cat = []

        if 'faceAnnotations' in file_keys:
            file_face = file['faceAnnotations']
            n_faces = len(file_face)
        else:
            n_faces = 0

        if 'textAnnotations' in file_keys:
            text_annotations = file['textAnnotations']
            file_n_text_annotations = len(text_annotations)
            file_len_text = [len(text['description']) for text in text_annotations]
        else:
            file_n_text_annotations = 0
            file_len_text = []

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = [x['score'] for x in file_colors]
        file_color_pixelfrac = [x['pixelFraction'] for x in file_colors]
        file_color_red = [x['color']['red'] if 'red' in x['color'].keys() else 0 for x in file_colors]
        file_color_blue = [x['color']['blue'] if 'blue' in x['color'].keys() else 0 for x in file_colors]
        file_color_green = [x['color']['green'] if 'green' in x['color'].keys() else 0 for x in file_colors]
        file_crop_conf = np.mean([x['confidence'] for x in file_crops])
        file_crop_x = np.mean([x['boundingPoly']['vertices'][1]['x'] for x in file_crops])
        file_crop_y = np.mean([x['boundingPoly']['vertices'][3]['y'] for x in file_crops])

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.mean([x['importanceFraction'] for x in file_crops])
        else:
            file_crop_importance = 0

        metadata = {
            'annots_top_desc': ' '.join(file_top_desc),
            'n_faces': n_faces,
            'n_text_annotations': file_n_text_annotations,
            'crop_conf': file_crop_conf,
            'crop_x': file_crop_x,
            'crop_y': file_crop_y,
            'crop_importance': file_crop_importance,
        }
        metadata.update(self.get_stats(file_top_score, 'annots_score'))
        metadata.update(self.get_stats(file_color_score, 'color_score'))
        metadata.update(self.get_stats(file_color_pixelfrac, 'color_pixel_score'))
        metadata.update(self.get_stats(file_color_red, 'color_red_score'))
        metadata.update(self.get_stats(file_color_blue, 'color_blue_score'))
        metadata.update(self.get_stats(file_color_green, 'color_green_score'))
        metadata.update(self.get_stats(dog_cat_scores, 'dog_cat_scores'))
        metadata.update(self.get_stats(dog_cat_topics, 'dog_cat_topics'))
        metadata.update(self.get_stats(is_dog_or_cat, 'is_dog_or_cat'))
        metadata.update(self.get_stats(file_len_text, 'len_text'))

        return pd.Series(metadata)

    def _transform(self, path, sentiment=True):
        file = self.open_json_file(path)
        if sentiment:
            result = self.parse_sentiment_file(file)
        else:
            result = self.parse_metadata(file)
        return result


def len_text_features(train):
    train['Length_Description'] = train['Description'].map(len)
    train['Length_annots_top_desc'] = train['annots_top_desc'].map(len)
    train['Lengths_sentences_text'] = train['sentences_text'].map(len)

    return train


with timer('merge additional files'):
    train = merge_breed_name(train)

with timer('metadata'):
    # TODO: parallelization
    meta_parser = MetaDataParser()
    sentiment_features = meta_parser.sentiment_files['sentiment_filename'].apply(
        lambda x: meta_parser._transform(x, sentiment=True))
    meta_parser.sentiment_files = pd.concat([meta_parser.sentiment_files, sentiment_features], axis=1, sort=False)
    meta_features = meta_parser.metadata_files['metadata_filename'].apply(
        lambda x: meta_parser._transform(x, sentiment=False))
    meta_parser.metadata_files = pd.concat([meta_parser.metadata_files, meta_features], axis=1, sort=False)

    stats = ['mean', 'min', 'max', 'median']
    columns = [c for c in sentiment_features.columns if c != 'sentiment_text']
    g = meta_parser.sentiment_files[list(sentiment_features.columns) + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

    columns = [c for c in meta_features.columns if c != 'annots_top_desc']
    g = meta_parser.metadata_files[columns + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

with timer('metadata, annots_top_desc'):
    meta_features = meta_parser.metadata_files[['PetID', 'annots_top_desc']]
    meta_features = meta_features.groupby('PetID')['annots_top_desc'].sum().reset_index()
    train = train.merge(meta_features, how='left', on='PetID')

    sentiment_features = meta_parser.sentiment_files[['PetID', 'sentiment_text']]
    sentiment_features = sentiment_features.groupby('PetID')['sentiment_text'].sum().reset_index()
    train = train.merge(sentiment_features, how='left', on='PetID')

    train['desc'] = ''
    for c in ['BreedName_main_breed', 'BreedName_second_breed', 'annots_top_desc', 'sentences_text']:
        train['desc'] += ' ' + train[c].astype(str)


with timer('kernel text features'):
    orig_cols = train.columns
    train = len_text_features(train)
    new_cols = [c for c in train.columns if c not in orig_cols]
    train[new_cols].to_feather("../feature/kernel_text.feather")