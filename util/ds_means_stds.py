import torch, time

def mean_stds(dataset):

    # determine dataset means and stds
    def calc():

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=48, shuffle=True, num_workers=0
        )

        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for images, _, _, _ in loader:
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                        dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
        return mean, std

    start_time = time.time()
    print("Calculating global means and standards. This may take a while...")
    mean, std = calc()
    time_elapsed = time.time() - start_time
    print('Calculated global means and standards in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Dataset (GLOBAL) mean and std: \n", mean, std)

    return mean, std