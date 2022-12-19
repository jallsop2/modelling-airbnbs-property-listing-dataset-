from PIL import Image
import os
from math import floor


def resize_image(image, min_height):
    

    new_width = floor(image.width*(min_height/image.height))

    resized_image = image.resize((new_width,min_height))
    
    if resized_image.height != min_height:
        print(resized_image.height)

    return resized_image


def prepare_image_data():

    if not os.path.isdir("processed_images"):
        os.makedirs("processed_images")

    dirs =  os.listdir('images')
    raw_image_data = []

    for dir in dirs:
        prop_image_list = os.listdir(f'images/{dir}')

        for name in prop_image_list:
            raw_image_data.append(Image.open(f'images/{dir}/{name}'))

    min_height = min([image.height for image in raw_image_data if image.mode == 'RGB'])
    num_images = len(raw_image_data)

    raw_image_data.clear()

    counter = 0
    
    for dir in dirs:

        if not os.path.isdir(f'processed_images/{dir}'):
            os.mkdir(f'processed_images/{dir}')

        prop_image_list = os.listdir(f'images/{dir}')

        for image_name in prop_image_list:

            counter += 1
            print(f'{counter}/{num_images}', end='\r')

            image = Image.open(f'images/{dir}/{image_name}')

            if image.mode != 'RGB':
                continue
      
            image = resize_image(image, min_height)
        
            image.save(f'processed_images/{dir}/{image_name}')


if __name__ == '__main__':

    prepare_image_data()





