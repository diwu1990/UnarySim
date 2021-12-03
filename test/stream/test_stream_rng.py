import torch
from UnarySim.stream import RNG

def test_rng():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hwcfg = {
        "width" : 4, 
        "dim" : 1, 
        "rng" : "sobol"
    }
    swcfg={
        "rtype" : torch.float
    }
    rng = RNG(hwcfg, swcfg)()
    assert rng is not None
    print(hwcfg["rng"], rng.to(device))

    hwcfg["rng"] = "race"
    rng = RNG(hwcfg, swcfg)()
    assert rng is not None
    print(hwcfg["rng"], rng.to(device))

    hwcfg["rng"] = "lfsr"
    rng = RNG(hwcfg, swcfg)()
    assert rng is not None
    print(hwcfg["rng"], rng.to(device))

    hwcfg["rng"] = "sys"
    rng = RNG(hwcfg, swcfg)()
    assert rng is not None
    print(hwcfg["rng"], rng.to(device))


if __name__ == '__main__':
    test_rng()