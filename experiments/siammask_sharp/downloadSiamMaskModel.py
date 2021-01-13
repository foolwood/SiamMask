import requests
import os
#? Basic Example

# url = 'https://www.facebook.com/favicon.ico'
# r = requests.get(url, allow_redirects=True)

# open('facebook.ico', 'wb').write(r.content)

#? Basic Example

#* Running Part

base_url = 'http://www.robots.ox.ac.uk/~qwang/'

urlArray = [
    "SiamMask_VOT.pth",
    "SiamMask_VOT_LD.pth",
    "SiamMask_DAVIS.pth"
]

for url in urlArray:
    print("Pretrained models are downloading...")
    if not os.path.exists(url):
        r = requests.get(base_url + url, allow_redirects = True)
        open(url, 'wb').write(r.content)

print("Download complete...")