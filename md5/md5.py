# Created by Tremeschin (https://github.com/tremeschin)
# To obtain the 2 MD5 Values

import hashlib

def md5_parts(string):
    hash_bytes = hashlib.md5(string.encode('utf-8')).digest()
    low = int.from_bytes(hash_bytes[0:8], 'big', signed=True)
    high = int.from_bytes(hash_bytes[8:16], 'big', signed=True)
    print(f"0x{low & 0xffffffffffffffff:016x}")
    print(f"0x{high & 0xffffffffffffffff:016x}")

md5_parts("minecraft:blocks/acacia_button") # Change this