# Cargar experimentos
Para cargar un cubo resultado de una sesion de captura se debe usar el script load_experiments.py.
Este programa permite para visualizar el espectro de un pixel. Para correr:

```bash
    python load_experiments.py path_to_experiment px py min_wv max_wv
```

Donde:
    px: columnas
    py: filas
    min_wv : longitud de onda minima
    max_wv: longitud de onda maxima

min_wv debe ser minimo 387nm y max_wv debe ser maximo 1020nm
