#https://www.kaggle.com/keras/inceptionresnetv2
from utils import *

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Conv1D

def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im


def load_image(path):
    image = cv2.imread(path)
    #new_image = resize_to_square(image)
    new_image = cv2.resize(image, (inp[0][0], inp[0][1]))
    new_image = preprocess_input(new_image)
    return new_image

with timer('image'):
    train_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_images/*.jpg'))
    image_files = train_image_files + test_image_files
    train_images = pd.DataFrame(image_files, columns=['image_filename'])
    train_images['PetID'] = train_images['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

with timer('densenet'):
    batch_size = 16
    pet_ids = train_images['PetID'].values
    img_pathes = train_images['image_filename'].values
    n_batches = len(pet_ids) // batch_size + 1

    inp = Input((256, 256, 3))
    backbone = InceptionResNetV2(input_tensor=inp,
                   weights='../../input/Inception-Resnet-V2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)
    m = Model(inp, out)
    m.summary()

    features = []
    for b in range(n_batches):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = pet_ids[start: end]
        batch_path = img_pathes[start: end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i,(pet_id, path) in enumerate(zip(batch_pets, batch_path)):
            try:
                batch_images[i] = load_image(path)
            except:
                try:
                    batch_images[i] = load_image(path)
                except:
                    pass
        batch_preds = m.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features.append([pet_id] + list(batch_preds[i]))
    X = pd.DataFrame(features, columns=["PetID"]+["inceptionresnetv2_{}".format(i) for i in range(batch_preds.shape[1])])
    gp = X.groupby("PetID").mean().reset_index()
    #train = pd.merge(train, gp, how="left", on="PetID")
    del m; gc.collect()
    pd.merge(train[["PetID"]], gp, how="left", on="PetID").drop("PetID", axis=1).to_feather("../feature/inceptionresnetv2.feather")