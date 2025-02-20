import random
import torch
import torch.nn.functional as F

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if x.shape[1] > max_len:
            raise ValueError("not max_len")
        if x.dim() == 3:
            x = x.transpose(1, 2)
        x_padded = F.pad(
            x, (0, max_len - x.shape[-1]), mode="constant", value=PAD
        )
        if x.dim() == 3:
            x_padded = x_padded.transpose(1, 2)
        assert x.shape[0] == x_padded.shape[0] and (x.shape[2] == x_padded.shape[2] if x.dim == 3 else True)
        return x_padded

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.shape[1] for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


class AudioPool():
    """This class implements an audio buffer that stores previously generated audios.

    This buffer enables us to update discriminators using a history of generated audios
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the AudioPool class

        Parameters:
            pool_size (int) -- the size of audio buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_audios = 0
            self.audios = []
            self.audio_lens = []

    def query(self, audios, audio_lens):
        """Return an audio from the pool.

        Parameters:
            audios: the latest generated audios from the generator

        Returns audios from the buffer.

        By 50/100, the buffer will return input audios.
        By 50/100, the buffer will return audios previously stored in the buffer,
        and insert the current audios to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return audios, audio_lens
        return_audios = []
        return_audio_lens = []
        for audio, audio_len in zip(audios, audio_lens):
            audio = torch.squeeze(audio, 0)  # (C, L, ...)
            audio_len = torch.squeeze(audio_len, 0)  # (1,)
            if self.num_audios < self.pool_size:   # if the buffer is not full; keep inserting current audios to the buffer
                self.num_audios = self.num_audios + 1
                self.audios.append(audio)
                self.audio_lens.append(audio_len)
                return_audios.append(audio)
                return_audio_lens.append(audio_len)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored audio, and insert the current audio into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp_audio = self.audios[random_id].clone()
                    tmp_audio_len = self.audio_lens[random_id].clone()
                    self.audios[random_id] = audio
                    self.audio_lens[random_id] = audio_len
                    return_audios.append(tmp_audio)
                    return_audio_lens.append(tmp_audio_len)
                else:       # by another 50% chance, the buffer will return the current audio
                    return_audios.append(audio)
                    return_audio_lens.append(audio_len)
        # return_audios = torch.cat(return_audios, 0)   # collect all the audios and return
        return_audios = pad_2D(return_audios)  # [(C, L_i, ...) for _ in range(B)] -> (B, C, L_max, ...)
        return return_audios, return_audio_lens
