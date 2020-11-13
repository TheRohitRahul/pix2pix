from glob import glob
import os
import cv2
import torch
from random import randint
import numpy as np
from PIL import Image
import skimage

from unet_utils import rotate_bound

'''
The name of the image in A folder and B folder is the same
This loader is to convert image from domain A to image from domain B
'''
class ABLoader(object):
    def __init__(self, A_folder, B_folder, should_augment = True, fixed_height = 512, fixed_width = 512):

        self.should_augment = should_augment

        self.fixed_height = fixed_height
        self.fixed_width = fixed_width

        self.Afolder = A_folder
        self.B_folder = B_folder

        self.A_images = glob(A_folder + "/*.png")
        self.A_images.extend( glob(A_folder + "/*.jpg") )
        self.A_images.extend( glob(A_folder + "/*.jpeg") )


        # This is just for checking if the length of images in both the folders is the same
        self.B_images = glob(B_folder + "/*.png")
        self.B_images.extend( glob(A_folder + "/*.jpg") )
        self.B_images.extend( glob(A_folder + "/*.jpeg") )

        assert len(self.A_images) == len(self.B_images), "A and B folder are not of the same length hence exiting\n {} != {}".format(len(self.A_images), len(self.B_images))
        assert len(self.A_images) != 0, "Did not find any images in any folder. Check the folder name and the extension of the images, supported extensions are : png jpg and jpeg"

    def __len__(self):
        return len(self.A_images)
    
    def add_image_noise(self, image):
        pil_image = Image.fromarray(image)
        
        noise_array = glob(os.path.join(os.getcwd(), "noise") + "/*.png")
        
        if len(noise_array) == 0:
            return image

        select_noise = randint(0, len(noise_array) -1 )
        noise_image_path = noise_array[select_noise]
        
        noise_image = Image.open(noise_image_path)
        
        rotation_array_noise = [0, 90, 180, 270]
        rotation_select = randint(0, len(rotation_array_noise) - 1)
        
        noise_image = noise_image.rotate(rotation_array_noise[rotation_select])

        pil_image.paste(noise_image, (0, 0), noise_image)
         
        open_cv_image = np.array(pil_image) 
        # print(open_cv_image.shape)
        # exit()
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image
    
    def add_noise(self, image):
        
        noise_selection = randint(0,6)
        type_noise = "gaussian"

        if (noise_selection == 1):
            type_noise = "localvar"

        elif(noise_selection == 2):
            type_noise = "poisson"
        
        elif(noise_selection == 3):
            type_noise = "salt"
        
        elif(noise_selection == 4):
            type_noise = "pepper"

        elif(noise_selection == 5):
            type_noise = "s&p"
        
        elif (noise_selection == 6):
            type_noise = "speckle"

        image = skimage.img_as_ubyte(skimage.util.random_noise(image, mode=type_noise, seed=None, clip=True))
        return image
   
    '''
    This function will handle all the augmentation that will be applied to the image
    '''
    def augment(self, A_image, B_image):

        if not randint(0,20) :
            A_image = self.add_noise(A_image)
        
        if randint(0,1):
            angle = randint(0, 180) - 90
            A_image = rotate_bound(A_image, angle)
            B_image = rotate_bound(B_image, angle)

        if randint(0,1):
            kernel = np.ones((3,3))
            A_image = cv2.erode(A_image, kernel, iterations = 1)  

        return A_image, B_image

    def resize_image(self, A_image, B_image):
        
        resize_x = randint(4, 40)/10.0
        resize_y = resize_x#randint(4, 20)/10.0
        
        A_image = cv2.resize(A_image, dsize=None, fx=resize_x, fy=resize_y, interpolation = cv2.INTER_AREA)
        B_image = cv2.resize(B_image, dsize=None, fx=resize_x, fy=resize_y, interpolation = cv2.INTER_AREA)

        return A_image, B_image

    '''
    This function will handle what to do if sizes of the input image is different than the size that has been fixed
    '''
    def handle_size_issues(self, A_image, B_image):
        image_height, image_width = A_image.shape[:2]
        
        # print("----")
        # print(image_height, image_width)
        
        start_point_height = 0
        end_point_height = max( image_height, self.fixed_height)
        start_point_width = 0
        end_point_width = max(image_width, self.fixed_width)

        # Case 1 : image dim is greater 
        if image_height > self.fixed_height:
            start_point_height = randint(0, image_height - self.fixed_height)
            end_point_height = start_point_height + self.fixed_height
        else:
            diff_height = self.fixed_height - image_height 
            A_image = cv2.copyMakeBorder(A_image, 0, diff_height, 0, 0, cv2.BORDER_CONSTANT, value=255)
            B_image = cv2.copyMakeBorder(B_image, 0, diff_height, 0, 0, cv2.BORDER_CONSTANT, value=255)

        if image_width > self.fixed_width:
            start_point_width = randint(0, image_width - self.fixed_width)
            end_point_width = start_point_width + self.fixed_width

        else:
            diff_width = self.fixed_width - image_width 
            A_image = cv2.copyMakeBorder(A_image, 0, 0, 0, diff_width, cv2.BORDER_CONSTANT, value=255)
            B_image = cv2.copyMakeBorder(B_image, 0, 0, 0, diff_width, cv2.BORDER_CONSTANT, value=255)
        
        
        A_image = A_image[start_point_height : end_point_height, start_point_width : end_point_width]
        B_image = B_image[start_point_height : end_point_height, start_point_width : end_point_width]

        # print(A_image.shape[0], A_image.shape[1])
        # print(B_image.shape[0], B_image.shape[1])
        # print("after handle function----")
        # if the input image is bigger than the 
        return A_image, B_image

    def debug_image(self, image_heading, image):
        cv2.imshow(image_heading, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cropped_image_specific_augmentations(self, A_image, B_image):
        A_image = self.add_image_noise(A_image)
        
        # if randint(0,1):
        #     A_image = 255 - A_image
        #     B_image = 255 - B_image
        
        kernel = randint(2, 4)
        A_image = cv2.blur(A_image,(kernel, kernel))   
        
        
        return A_image, B_image

    def __getitem__(self, i, debug=False):
        
        A_image_path = self.A_images[i]
        image_name = os.path.basename(A_image_path)
        B_image_path= os.path.join( self.B_folder, image_name )

        # print("A image path is {}\nB image path is {}".format(A_image_path, B_image_path))
        
        A_image = cv2.imread(A_image_path, 0)
        B_image = cv2.imread(B_image_path, 0)
        

        if self.should_augment:
            A_image, B_image = self.augment(A_image, B_image)

        if randint(0,1):
            A_image, B_image = self.resize_image(A_image, B_image)
        
        
        A_image, B_image = self.handle_size_issues(A_image, B_image)
        
        A_image, B_image = self.cropped_image_specific_augmentations(A_image, B_image)
        
        
        A_image = A_image / 255.0
        B_image = B_image / 255.0

        
        # if debug:
        #     self.debug_image("A", A_image)
        #     self.debug_image("B", B_image)

        # print(A_image.shape[0], A_image.shape[1])
        # print(B_image.shape[0], B_image.shape[1])
        # print("before tensor conversion ----")
        A_image = torch.Tensor(A_image).unsqueeze(0)
        B_image = torch.Tensor(B_image).unsqueeze(0)

        # print("after tensor conversion")
        # print(A_image.size())
        # print(B_image.size())
        return A_image, B_image


if __name__ == "__main__":
    A_image_folder = ""
    B_image_folder = ""
    
    ncl_obj = ABLoader(A_image_folder, B_image_folder)
    i = randint(0, len(ncl_obj)-1) 
    ncl_obj.__getitem__(i, debug=True)