import random
import torch
import torch.nn.functional as F

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        s = x.shape[1]
        x_padded = F.pad(
            x.transpose(0, 1), (0, max_len - x.shape[0]), mode="constant", value=PAD
        ).transpose(0, 1)
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


class AudioPool():
    """This class implements an image buffer that stores previously generated audios.

    This buffer enables us to update discriminators using a history of generated audios
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_audios = 0
            self.audios = []

    def query(self, audios):
        """Return an image from the pool.

        Parameters:
            audios: the latest generated audios from the generator

        Returns audios from the buffer.

        By 50/100, the buffer will return input audios.
        By 50/100, the buffer will return audios previously stored in the buffer,
        and insert the current audios to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return audios
        return_audios = []
        for audio in audios:
            audio = torch.squeeze(audio.data, 0)
            if self.num_audios < self.pool_size:   # if the buffer is not full; keep inserting current audios to the buffer
                self.num_audios = self.num_audios + 1
                self.audios.append(audio)
                return_audios.append(audio)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.audios[random_id].clone()
                    self.audios[random_id] = audio
                    return_audios.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_audios.append(audio)
        # return_audios = torch.cat(return_audios, 0)   # collect all the audios and return
        return_audios = pad_2D(return_audios)
        return return_audios
