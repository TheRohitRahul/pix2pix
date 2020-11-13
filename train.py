import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import torchvision.utils as vutils
from tqdm.auto import tqdm

from patchgan_discriminator import PatchGANDiscriminator
from unet_generator import UnetGenerator

from unet_utils import write_image, init_weights
from BatchLoader import BatchLoader
from config import INPUT_CHANNELS, OUTPUT_CHANNELS, PRINT_EVERY,\
     NUM_ITERATIONS, DEVICE, DATASET_DICT, BATCH_SIZE, MODEL_SAVE_PATH,\
         TRAIN_RESUME_PATH

def get_generator_loss(gen, disc, loss_criteria_gan, l1_loss_criteria, real_input, real_output, lambda_param = 100):
    
    gen_fake_output = gen(real_input)
    disc_pred_fake = disc(torch.cat([real_output, gen_fake_output], dim = 1))
    
    ground_truth = torch.ones_like(disc_pred_fake)
    
    gan_loss = loss_criteria_gan(disc_pred_fake, ground_truth)
    l1_loss = l1_loss_criteria(real_output, gen_fake_output)

    gen_loss = gan_loss + lambda_param*l1_loss

    return gen_loss


def get_discriminator_loss(gen, disc, loss_criteria_gan, real_input, real_output):
    gen_fake_output = gen(real_input).detach()
    disc_pred_fake = disc(torch.cat([real_output , gen_fake_output], dim = 1))
    disc_pred_real = disc(torch.cat([real_output, real_input], dim = 1))

    gt_fake = torch.zeros_like(disc_pred_fake)
    gt_real = torch.ones_like(disc_pred_real)

    fake_loss = loss_criteria_gan(disc_pred_fake, gt_fake)
    real_loss = loss_criteria_gan(disc_pred_real, gt_real)

    disc_loss = (fake_loss + real_loss)*0.5
    return disc_loss

def train(dataset_dict = DATASET_DICT, model_path = ""):


    train_obj = BatchLoader(dataset_dict=dataset_dict, batch_size = BATCH_SIZE)
    
    discriminator = PatchGANDiscriminator(INPUT_CHANNELS + OUTPUT_CHANNELS).to(DEVICE)
    generator = UnetGenerator(INPUT_CHANNELS, OUTPUT_CHANNELS).to(DEVICE)

    if model_path == "":
        print("Initializing weights for training from scratch")
        init_weights(discriminator)
        init_weights(generator)
    else:
        print("Loading from model path : {}".format(model_path))
        model = torch.load(model_path)
        discriminator.load_state_dict(model["discriminator"])
        generator.load_state_dict(model["generator"])

    optimizer_gen = Adam(generator.parameters())
    optimizer_disc = Adam(discriminator.parameters())
    l1_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    
    disc_loss_arr = []
    gen_loss_arr = []
    for iteration_index in tqdm(range(NUM_ITERATIONS)):
        discriminator.train()
        generator.train()

        real_input, real_output = train_obj.get_batch()
        real_input, real_output = real_input.to(DEVICE), real_output.to(DEVICE)

        optimizer_disc.zero_grad()
        disc_loss = get_discriminator_loss(generator, discriminator, gan_loss, real_input, real_output)
        disc_loss.backward()
        optimizer_disc.step()
        disc_loss_arr.append(disc_loss.item())

        optimizer_gen.zero_grad()
        gen_loss = get_generator_loss(generator, discriminator, gan_loss, l1_loss, real_input, real_output)
        gen_loss.backward()
        optimizer_gen.step()
        gen_loss_arr.append(gen_loss.item())
        

        if (iteration_index%PRINT_EVERY == 0):
            
            mean_gen_loss, mean_disc_loss = np.mean(gen_loss_arr), np.mean(disc_loss_arr)
            disc_loss_arr = []
            gen_loss_arr = []

            print("Iteration : {}\nGenerator Loss : {}\n Discriminator Loss : {}".format(iteration_index, mean_gen_loss, mean_disc_loss))
            
            dict_to_save = {
                "discriminator" : discriminator.state_dict(),
                "generator" : generator.state_dict()
            }
            save_model_path = os.path.join(MODEL_SAVE_PATH, "iteration_{}_gLoss_{:.4f}_dLoss_{:.4f}.pt".format(iteration_index, mean_gen_loss, mean_disc_loss))
            torch.save(dict_to_save, save_model_path)
            
            fake_out = None
            with torch.no_grad():
                fake_out = generator(real_input).detach().cpu()

            write_image(vutils.make_grid(fake_out, padding=2, normalize=True), "{}_fake".format(iteration_index))
            write_image(vutils.make_grid(real_input, padding=2, normalize=True), "{}_input".format(iteration_index))
            write_image(vutils.make_grid(real_output, padding=2, normalize=True), "{}_gt".format(iteration_index))


if __name__ == "__main__":
    train(model_path=TRAIN_RESUME_PATH)