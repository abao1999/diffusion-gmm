from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from diffusion_gmm.gmm import ImageGMM


WORK_DIR = os.getenv('WORK')
DATA_DIR = os.path.join(WORK_DIR, 'vision_datasets')


if __name__ == '__main__':

    use_generated_data = False
    batch_size = 64

    # for gmm fitting
    n_for_stats = 1024
    n_for_fit = 1024

    # for generating samples
    dataset_name = 'cifar10'
    cifar10_shape = (3, 32, 32)
    n_samples_generate = 1024

    # Define the transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the dataset
    if use_generated_data:
        # load generated images
        image_dir = os.path.join(DATA_DIR, f"diffusion_{dataset_name}")
        dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    else:
        # load real images
        image_dir = os.path.join(DATA_DIR, dataset_name)
        dataset = datasets.CIFAR10(
            root=os.path.join(DATA_DIR, 'cifar10'), 
            train=False, 
            download=True, 
            transform=transform
        )

        # # Limit the number of samples
        # num_samples = 1000
        # dataset.data = dataset.data[:num_samples]
        # dataset.targets = dataset.targets[:num_samples]

    print("Image directory:", image_dir)

    n_tot_images = len(dataset)
    if n_for_stats > n_tot_images:
        print(f"Warning: Only {n_tot_images} images found in the dataset. Using all available images.")
        n_for_stats = n_tot_images

    custom_sampler = SubsetRandomSampler(range(n_for_stats))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        sampler=custom_sampler # if custom_sampler else None
    )

    gmm = ImageGMM(
        dataloader=dataloader,
        img_shape=cifar10_shape, 
        n_components=10, 
        verbose=True,
    )

    # gmm(n_samples_compute_stats=1024, n_samples_fit=1024, run_name='gmm_cifar10')

    gmm.fit(n_samples_compute_stats=n_for_stats, n_samples_fit=n_for_fit)
    print("GMM fitted successfully.")

    # # Save the generated images
    # gmm.save_samples(
    #     n_samples=100, #n_samples_gmm, 
    #     save_fig_dir='figs',
    #     save_grid_shape=(10, 10),
    #     save_name=f"gmm_{dataset_name}",
    # )

    save_dir = os.path.join(DATA_DIR, 'gmm_cifar10', 'unknown')
    gmm.save_samples(
        n_samples=n_samples_generate, 
        save_fig_dir=save_dir,
        save_grid_shape=None,
        save_name=f"gmm_{dataset_name}",
    )
    
    # # Generate samples from the fitted GMM
    # print(f"Generating samples from the fitted GMM...")
    # samples, _ = gmm.sample(n_samples_gmm)
    # # Plot the histogram of samples generated from the fitted GMM
    # plot_pixel_intensity_hist(samples, bins=100)
