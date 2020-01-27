import numpy as np
import glob
from tqdm import tqdm

# paths = glob.glob("/srv/share/ykant3/vilbert-mt/features/obj/test/*") # clean
# paths = glob.glob("/srv/share/ykant3/vilbert-mt/features/ocr/test/*") # clean
paths = glob.glob("/srv/share/ykant3/vilbert-mt/features/ocr/train/*") # clean


# paths = glob.glob("/srv/share/ykant3/vilbert-mt/features/obj/train/*") # running (has problems)
for path in tqdm(paths):
    try:
        data = np.load(path, allow_pickle=True).item()
        assert "ocr_boxes" in data
    except BaseException:
        print(path)

print("Done!")