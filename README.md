# find-duplicate-images

Find exact and near-duplicate images in a folder. Prints image titles (EXIF or filename) and can write a CSV report.

Usage example:

```pwsh
python find-duplicate-images.py C:\path\to\images --phash-threshold 6 --out duplicates.csv
```

Dependencies: pillow, imagehash, tqdm
