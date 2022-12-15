from PIL import Image
import os
from math import floor


def read_image_data():

    print('\nReading image data: ', end="")

    dirs =  os.listdir('images')
    

    raw_image_data = {}

    
    for dir in dirs:
        image_name_list = os.listdir(f'images/{dir}')

        for name in image_name_list:
            raw_image_data[name] = Image.open(f'images/{dir}/{name}')
        
    print(f'{len(raw_image_data)} total images')

    return raw_image_data


def resize_images(image_data):

    print('\nResizing images:')

    min_height = min([image.height for image in image_data.values() if image.mode == 'RGB'])

    counter = 0
    num_images = len(image_data)
    
    for image_name, image in image_data.items():

        counter += 1
        print(f'{counter}/{num_images}', end='\r')

        if image.mode != 'RGB':
            continue

        new_width = floor(image.width*(min_height/image.height))

        resized_image = image.resize((new_width,min_height))
        
        if resized_image.height != min_height:
            print(resized_image.height)

        image_data[image_name] = resized_image

    return image_data


def save_image_data(image_data):

    print('\n\nSaving images')

    if not os.path.isdir("processed_images"):
        os.makedirs("processed_images")

    counter = 0
    num_images = len(image_data)

    for image_name, image in image_data.items():

        counter += 1
        print(f'{counter}/{num_images}', end='\r')

        prop_id = image_name[:image_name.rindex('-')]

        if not os.path.isdir(f'processed_images/{prop_id}'):
            os.mkdir(f'processed_images/{prop_id}')
        
        image.save(f'processed_images/{prop_id}/{image_name}')


def prepare_image_data():

    image_data = read_image_data()

    resized_image_data = resize_images(image_data)

    save_image_data(resized_image_data)


if __name__ == '__main__':

    prepare_image_data()



