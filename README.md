# DTI
Esta API, esta diseñada para predecir la interacción entre un farmaco(representado en una cadena SMILES) y una proteina(representado en FASTA). 

* 0 interactua. 
* 1 No interactua.

# Requerimientos

# Instalación
Para evitar problemas en la instalación, hicimos un contenedor para correr la API:

```bash
  docker run -it 11treu11/dti -p 8000:8000 --gpus all /bin/bash
```
Imagen 1

```bash
  cd
  cd DTI/API
  conda activate DTI
  python API.py
```
Imagen2
