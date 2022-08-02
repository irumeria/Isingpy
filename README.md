# Isingpy

## Discription
A python implement for **ising model** and **metropolis algorithm** in the book <a href="https://www.cambridge.org/highereducation/books/introduction-to-computational-materials-science/327CCEC340E5C466CE08D6A6FD8520E1#overview" >introduction to computational materials science</a>(Chapter 7)

## Usage
+ Install the nessary dependancies
  + pytorch
+ Config can be set in hparams.py
```python
    {
        "points_number": 10,
        "epochs": 1000,
        "gpu": False,
    }
```
+ Run the code
```bash
python ./src/ising.py
```

## Reference
+ About the <a href="https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm">metropolis algorithm</a>
+ About the <a href="https://en.wikipedia.org/wiki/Ising_model">ising model</a>