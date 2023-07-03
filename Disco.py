import torch

def distance_corr(power, var_1, var_2, normedweight):
    """
    From https://github.com/okitouni/DisCo
    var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    xx = var_1.view(-1, 1).expand(len(var_1), len(var_1)).view(len(var_1), len(var_1))
    yy = var_1.expand(len(var_1), len(var_1)).view(len(var_1), len(var_1))
    amat = (xx-yy).abs()
    del xx, yy

    amatavg = torch.mean(amat*normedweight, dim=1)
    Amat = amat-amatavg.expand(len(var_1), len(var_1)).view(len(var_1), len(var_1))\
        - amatavg.view(-1, 1).expand(len(var_1), len(var_1)).view(len(var_1), len(var_1))\
        + torch.mean(amatavg*normedweight)
    del amat

    xx = var_2.view(-1, 1).expand(len(var_2), len(var_2)).view(len(var_2), len(var_2))
    yy = var_2.expand(len(var_2), len(var_2)).view(len(var_2), len(var_2))
    bmat = (xx-yy).abs()
    del xx, yy

    bmatavg = torch.mean(bmat*normedweight, dim=1)
    Bmat = bmat-bmatavg.expand(len(var_2), len(var_2)).view(len(var_2), len(var_2))\
        - bmatavg.view(-1, 1).expand(len(var_2), len(var_2)).view(len(var_2), len(var_2))\
        + torch.mean(bmatavg*normedweight)
    del bmat

    ABavg = torch.mean(Amat * Bmat)
    AAavg = torch.mean(Amat * Amat)
    BBavg = torch.mean(Bmat * Bmat)
    del Bmat, Amat
    dCorr = (torch.sqrt(ABavg)) / torch.sqrt((torch.sqrt(AAavg) * torch.sqrt(BBavg)))

    return dCorr
