from PIL import Image
from utils import *


def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size


with timer('image'):
    train_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_images/*.jpg'))
    image_files = train_image_files + test_image_files
    train_images = pd.DataFrame(image_files, columns=['image_filename'])
    train_images['PetID'] = train_images['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

with timer('img_basic'):
    train_images['image_size'] = train_images['image_filename'].apply(getSize)
    train_images['temp_size'] = train_images['image_filename'].apply(getDimensions)
    train_images['width'] = train_images['temp_size'].apply(lambda x: x[0])
    train_images['height'] = train_images['temp_size'].apply(lambda x: x[1])
    train_images = train_images.drop(['temp_size'], axis=1)

    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var'],
    }

    gp = train_images.groupby('PetID').agg(aggs)
    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    gp.columns = new_columns
    gp = gp.reset_index()
    gp.drop("PetID", axis=1).to_feather("../feature/image_stats.feather")