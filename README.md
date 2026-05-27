# Rozpoznawanie liczb polskich - MFCC + DTW
 
## Wymagania
 
```bash
pip install -r requirements.txt
```
 
## Ewaluacja
 
**Tryb repetition** - nagrania `*_1.wav` jako trening, nagrania `*_2.wav` jako test:
```bash
python evaluate.py --split repetition
```
 
**Tryb speaker-independent** - wybrani mówcy jako test, pozostali jako trening:
```bash
python evaluate.py --split speaker --test-speakers 1 2 3 4
```
 