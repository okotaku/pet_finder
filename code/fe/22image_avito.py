from skimage import io
from skimage.color import gray2rgb
from utils import *


def calc_dom(X):
    img = Image.fromarray(X)
    img = np.float32(img)
    img = img.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(img, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quant = palette[labels.flatten()]
    quant = quant.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

    return dominant_color / 255


# average_color
def avg_c(X):
    average_color = [X[:, :, i].mean() for i in range(X.shape[-1])]
    return np.array(average_color) / 255


def calc_brightness(chennel, black_lim, white_lim):
    n_pixel = chennel.shape[0] * chennel.shape[1]
    whiteness = len(chennel[chennel > white_lim]) / n_pixel
    dullness = len(chennel[chennel < black_lim]) / n_pixel

    return whiteness, dullness


def average_pixel_width(im):
    im_array = np.asarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.shape[0] * im.shape[1]))
    return apw * 100


def get_blurrness_score(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


with timer('image'):
    train_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_images/*.jpg'))
    image_files = train_image_files + test_image_files
    train_images = pd.DataFrame(image_files, columns=['image_filename'])
    train_images['PetID'] = train_images['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

with timer('image'):
    train_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_images/*.jpg'))
    image_files = train_image_files + test_image_files
    train_images = pd.DataFrame(image_files, columns=['image_filename'])
    train_images['PetID'] = train_images['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

with timer('img_basic'):
    features = []
    for i, (file_path, pet_id) in enumerate(train_images.values):
        im = io.imread(file_path)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        avg = list(avg_c(im).flatten().astype(np.float32))

        color_list = ["red", "blue", "green"]
        whiteness_, dullness_ = [], []
        for i, color in enumerate(color_list):
            whiteness, dullness = calc_brightness(im[:, :, i], 20, 240)
            whiteness_ += [whiteness]
            dullness_ += [dullness]
        blu = get_blurrness_score(im)
        features.append(avg + whiteness_ + dullness_ + [blu, pet_id])

    color_list = ["red", "blue", "green"]
    cols = ["avg_" + color for color in color_list] + ["whiteness_" + color for color in color_list] + \
           ["dullness_" + color for color in color_list] + ["blurrness", "PetID"]
    X = pd.DataFrame(features, columns=cols)
    gp = X.groupby("PetID").mean().reset_index()
    new_cols = list(gp.drop("PetID", axis=1).columns)
    train = train.merge(gp, how="left", on="PetID")
    train[new_cols].to_feather("../feature/image_basic.feather")
