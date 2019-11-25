try:
    import kaggle
except:
    print('Please install the kaggle library')
    
def download_data (kaggle_json):
    os.environ['KAGGLE_USERNAME'] = kaggle_token['username'] # username from the json file 
    os.environ['KAGGLE_KEY'] = kaggle_token['key'] # key from the json file

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('brilja/pokemon-mugshots-from-super-mystery-dungeon',path = 'Data', unzip=True)

def load_data ():
    images = []

    for im_path in glob.glob("Data/smd/*.png"):
        images.append(imageio.imread(im_path))
    images = np.array(images)

    images = images / 255

    x_train, x_test = train_test_split(images, test_size = 0.1)

    return x_train, x_test

def augment_data (data, length):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
    )

    for i in datagen.flow(x_train, batch_size=length):
        x_train_aug = i
        break
    
    return x_train_aug