from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ProteinChain
import inspect

print("ESMProtein methods:", dir(ESMProtein))
print("ProteinChain methods:", dir(ProteinChain))
print("ESM3 forward signature:", inspect.signature(ESM3.forward))
