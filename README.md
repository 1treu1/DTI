# DTI
Esta API, esta diseñada para predecir la interacción entre un farmaco(representado en una cadena SMILES) y una proteina(representado en FASTA). 

* 0 interactua. 
* 1 No interactua.

# Requerimientos
* Tener GPU Nvidia
# Instalación
Para evitar problemas en la instalación, hicimos un contenedor para correr la API:

```bash
  sudo docker run -it -p 8000:8000 --gpus all 11treu11/dti /bin/bash
```
Imagen 1
Corriendo localmente:
```bash
  cd
  cd DTI/API
  conda activate DTI
  python API.py
```
Imagen2
* Abre el navegador y accede a esta dirección
```bash
http://127.0.0.1:8000
```
