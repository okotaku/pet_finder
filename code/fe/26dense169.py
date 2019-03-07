from utils import *


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
    backbone = DenseNet169(input_tensor=inp,
                   weights='../../input/densenet-keras/DenseNet-BC-169-32-no-top.h5',
                   include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)
    m = Model(inp, out)

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
    X = pd.DataFrame(features, columns=["PetID"]+["dense169_{}".format(i) for i in range(batch_preds.shape[1])])
    gp = X.groupby("PetID").mean().reset_index()
    #train = pd.merge(train, gp, how="left", on="PetID")
    del m; gc.collect()
    pd.merge(train[["PetID"]], gp, how="left", on="PetID").drop("PetID", axis=1).to_feather("../feature/dense169.feather")