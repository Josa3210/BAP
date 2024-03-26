import torch


class FeatureCacher:
    def cache(self, result, cache):
        torch.save(result, cache)

    def load(self, filePathCache) -> torch.Tensor:
        return torch.load(filePathCache)
