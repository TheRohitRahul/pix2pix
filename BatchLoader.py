import torch

from config import DATASET_DICT, NUM_WORKERS
from A_B_loader import ABLoader

class BatchLoader(object):
    def __init__(self, batch_size, dataset_dict = DATASET_DICT):
        
        self.dataset_list = []
        self.dataset_iter_list = []

        for a_dataset in dataset_dict:
            
            A_folder = dataset_dict[a_dataset]["A_folder"]
            B_folder = dataset_dict[a_dataset]["B_folder"]
            
            ratio = float(dataset_dict[a_dataset]["ratio"])
            batch_size_dataset = int(batch_size*ratio)

            print("{} will have a batch size of {}".format(a_dataset, batch_size_dataset))
            dataload_obj = ABLoader(A_folder, B_folder)
            print("{} will have a length of {}".format(a_dataset, len(dataload_obj)))
            dataload = torch.utils.data.DataLoader(dataload_obj, 
                        batch_size = batch_size_dataset,
                        shuffle=True,
                        num_workers=NUM_WORKERS)

            self.dataset_list.append(dataload)
            self.dataset_iter_list.append(iter(dataload))



    def __len__(self):
        return 10000

    def get_batch(self):
        A_images = []
        B_images = []

        for i, dataloader_iter in enumerate(self.dataset_iter_list):
            try:
                A_image, B_image = dataloader_iter.next()
                A_images.append(A_image)
                B_images.append(B_image)

            except StopIteration:
                self.dataset_iter_list[i] = iter(self.dataset_list[i])
                A_image, B_image = self.dataset_iter_list[i].next()
                A_images.append(A_image)
                B_images.append(B_image)

            except ValueError:
                print("valueerror")
                pass
        
        A_images = torch.cat(A_images, 0)
        B_images = torch.cat(B_images, 0)

        return A_images, B_images