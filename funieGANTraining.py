import os
import numpy as np
from modelNet.funieGAN import FUNIE_GAN
from utilityFiles.data_utils import DataLoader
from utilityFiles.plot_utils import save_val_samples_funieGAN

data_dir = "/content/drive/My Drive/Colab Notebooks/vikas/"
dataset_name = "underwater_imagenet" 
data_loader = DataLoader(os.path.join(data_dir, dataset_name), dataset_name)
samples_dir = os.path.join("data/samples/funieGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/funieGAN/", dataset_name)
if not os.path.exists(samples_dir): os.makedirs(samples_dir)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

num_epoch = 40
batch_size = 2
val_interval = 1000
N_val_samples = 1
save_model_interval = data_loader.num_train//batch_size
print(data_loader.num_train)
print(save_model_interval)
num_step = num_epoch*save_model_interval
print(num_step)

funie_gan = FUNIE_GAN()

valid = np.ones((batch_size,) + funie_gan.disc_patch)
fake = np.zeros((batch_size,) + funie_gan.disc_patch)


step = 0
all_D_losses = []; all_G_losses = []
while (step <= num_step):
    for _, (imgs_distorted, imgs_good) in enumerate(data_loader.load_batch(batch_size)):
        
        imgs_fake = funie_gan.generator.predict(imgs_distorted)
        
        funie_gan.discriminator.compile

        d_loss_real = funie_gan.discriminator.train_on_batch([imgs_good, imgs_distorted], valid)
        d_loss_fake = funie_gan.discriminator.train_on_batch([imgs_fake, imgs_distorted], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = funie_gan.combined.train_on_batch([imgs_good, imgs_distorted], [valid, imgs_good])
        step += 1; all_D_losses.append(d_loss[0]);  all_G_losses.append(g_loss[0]);
        if step%50==0:
            print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, num_step, d_loss[0], g_loss[0])) 
        if (step % val_interval==0):
            imgs_distorted, imgs_good = data_loader.load_val_data(batch_size=N_val_samples)
            imgs_fake = funie_gan.generator.predict(imgs_distorted)
            gen_imgs = np.concatenate([imgs_distorted, imgs_fake, imgs_good])
            gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to 0-1
            save_val_samples_funieGAN(samples_dir, gen_imgs, step, N_samples=N_val_samples)
        if (step % save_model_interval==0):
            model_name = os.path.join(checkpoint_dir, ("model_%d" %step))
            with open(model_name+"_.json", "w") as json_file:
                json_file.write(funie_gan.generator.to_json())
            funie_gan.generator.save_weights(model_name+"_.h5")
            print("\nThe Saved trained model is stored in {0}\n".format(checkpoint_dir))
        if (step>=num_step): break     

