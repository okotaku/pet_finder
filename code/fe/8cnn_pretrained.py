from keras import backend as K
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception, inception_v3
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras.models import Model
from utils import *


# with timer('vgg16'):
#     pet_ids = train['PetID'].values
#     n_batches = len(pet_ids) // batch_size + 1

#     inp = Input((256, 256, 3))
#     backbone = VGG16(input_tensor=inp,
#                   weights='../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                   include_top=False)
#     x = backbone.output
#     x = GlobalAveragePooling2D()(x)
#     x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
#     x = AveragePooling1D(4)(x)
#     out = Lambda(lambda x: x[:, :, 0])(x)
#     m = Model(inp, out)

#     features = {}
#     for b in range(n_batches):
#         start = b * batch_size
#         end = (b + 1) * batch_size
#         batch_pets = pet_ids[start: end]
#         batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
#         for i,pet_id in enumerate(batch_pets):
#             try:
#                 batch_images[i] = load_image('../input/petfinder-adoption-prediction/train_images/', pet_id)
#             except:
#                 try:
#                     batch_images[i] = load_image('../input/petfinder-adoption-prediction/test_images/', pet_id)
#                 except:
#                     pass
#         batch_preds = m.predict(batch_images)
#         for i, pet_id in enumerate(batch_pets):
#             features[pet_id] = batch_preds[i]
#     X = pd.DataFrame.from_dict(features, orient='index')
#     X = X.rename(columns=lambda i: f'vgg16_{i}').reset_index(drop=True)
#     train = pd.concat([train, X], axis=1)

# with timer('resnet50'):
#     pet_ids = train['PetID'].values
#     n_batches = len(pet_ids) // batch_size + 1

#     inp = Input((256, 256, 3))
#     backbone = ResNet50(input_tensor=inp,
#                   weights='../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                   include_top=False)
#     x = backbone.output
#     x = GlobalAveragePooling2D()(x)
#     x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
#     x = AveragePooling1D(4)(x)
#     out = Lambda(lambda x: x[:, :, 0])(x)
#     m = Model(inp, out)

#     features = {}
#     for b in range(n_batches):
#         start = b * batch_size
#         end = (b + 1) * batch_size
#         batch_pets = pet_ids[start: end]
#         batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
#         for i,pet_id in enumerate(batch_pets):
#             try:
#                 batch_images[i] = load_image('../input/petfinder-adoption-prediction/train_images/', pet_id)
#             except:
#                 try:
#                     batch_images[i] = load_image('../input/petfinder-adoption-prediction/test_images/', pet_id)
#                 except:
#                     pass
#         batch_preds = m.predict(batch_images)
#         for i, pet_id in enumerate(batch_pets):
#             features[pet_id] = batch_preds[i]
#     X = pd.DataFrame.from_dict(features, orient='index')
#     X = X.rename(columns=lambda i: f'resnet50_{i}').reset_index(drop=True)
#     train = pd.concat([train, X], axis=1)

# with timer('xception'):
#     pet_ids = train['PetID'].values
#     n_batches = len(pet_ids) // batch_size + 1

#     inp = Input((256, 256, 3))
#     backbone = xception.Xception(input_tensor=inp,
#                   weights='../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                   include_top=False)
#     x = backbone.output
#     x = GlobalAveragePooling2D()(x)
#     x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
#     x = AveragePooling1D(4)(x)
#     out = Lambda(lambda x: x[:, :, 0])(x)
#     m = Model(inp, out)

#     features = {}
#     for b in range(n_batches):
#         start = b * batch_size
#         end = (b + 1) * batch_size
#         batch_pets = pet_ids[start: end]
#         batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
#         for i,pet_id in enumerate(batch_pets):
#             try:
#                 batch_images[i] = load_image('../input/petfinder-adoption-prediction/train_images/', pet_id)
#             except:
#                 try:
#                     batch_images[i] = load_image('../input/petfinder-adoption-prediction/test_images/', pet_id)
#                 except:
#                     pass
#         batch_preds = m.predict(batch_images)
#         for i, pet_id in enumerate(batch_pets):
#             features[pet_id] = batch_preds[i]
#     X = pd.DataFrame.from_dict(features, orient='index')
#     X = X.rename(columns=lambda i: f'xception_{i}').reset_index(drop=True)
#     train = pd.concat([train, X], axis=1)

# with timer('inception'):
#     pet_ids = train['PetID'].values
#     n_batches = len(pet_ids) // batch_size + 1

#     inp = Input((256, 256, 3))
#     backbone = inception_v3.InceptionV3(input_tensor=inp,
#                   weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                   include_top=False)
#     x = backbone.output
#     x = GlobalAveragePooling2D()(x)
#     x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
#     x = AveragePooling1D(4)(x)
#     out = Lambda(lambda x: x[:, :, 0])(x)
#     m = Model(inp, out)

#     features = {}
#     for b in range(n_batches):
#         start = b * batch_size
#         end = (b + 1) * batch_size
#         batch_pets = pet_ids[start: end]
#         batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
#         for i,pet_id in enumerate(batch_pets):
#             try:
#                 batch_images[i] = load_image('../input/petfinder-adoption-prediction/train_images/', pet_id)
#             except:
#                 try:
#                     batch_images[i] = load_image('../input/petfinder-adoption-prediction/test_images/', pet_id)
#                 except:
#                     pass
#         batch_preds = m.predict(batch_images)
#         for i, pet_id in enumerate(batch_pets):
#             features[pet_id] = batch_preds[i]
#     X = pd.DataFrame.from_dict(features, orient='index')
#     X = X.rename(columns=lambda i: f'inception_v3_{i}').reset_index(drop=True)
#     train = pd.concat([train, X], axis=1)