# Initial Created by Tremeschin (github.com/tremeschin) where you input the resource string and it gives you the two MD5 Values

import asyncio
from playwright.async_api import async_playwright
import hashlib
from urllib.parse import urljoin

BASE_URL = "https://mcasset.cloud/1.21.10/data/minecraft/loot_table/gameplay/"
OUTPUT_FILE = "gameplay_md5s.txt"

def md5_parts(string: str):
    h = hashlib.md5(string.encode("utf-8")).digest()
    low = int.from_bytes(h[:8], "big", signed=True)
    high = int.from_bytes(h[8:], "big", signed=True)
    return (low & 0xffffffffffffffff, high & 0xffffffffffffffff)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(BASE_URL)
        
        await page.wait_for_selector("a")  
        
        hrefs = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
        json_files = [h for h in hrefs if h.endswith(".json")]

        json_files = sorted(set(json_files))  
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            for full_url in json_files:
                block_name = full_url.split("/")[-1][:-5] 
                namespaced = f"minecraft:gameplay/{block_name}"
                low, high = md5_parts(namespaced)

                out.write(f"{block_name}\n")
                out.write(f"0x{low:016x}\n")
                out.write(f"0x{high:016x}\n\n")

        print(f"Done! {len(json_files)} gameplay written to {OUTPUT_FILE}")
        await browser.close()

asyncio.run(main())