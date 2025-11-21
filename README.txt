
# Pré-requisitos
## Cria ambiente conda:
```bash
mamba create -n flim python=3.11 -y
mamba activate FLIM
```
## Instala IFT:
- configura a variável NEWIFT_DIR no seu ~/.bashrc
- instala cuda-toolkit, BLAS: 

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev
sudo apt install swig
sudo apt install nvidia-cuda-toolkit
```
### Compila IFT:
```bash
cd ift/
make clean
make
```

## Instala PyIFT
```bash
cd ift/PyIFT
pip install "numpy<2"
pip install -e .
```
Testa instalação:
```bash
python -c "import pyift.pyift"
```

## Instalação de Dependencias do Projeto
Instala PyTorch:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```
Instala PyFLIM:
```bash
cd flim-python-demo
pip install -e .
```

# Executando Experimentos
## Treinamento e Avaliação
```bash
python src/main.py fit -c configs/data/schisto.yaml -c configs/model/schisto_segmentation.yaml
python src/main.py test -c configs/data/schisto.yaml -c configs/model/schisto_segmentation.yaml --ckpt_path <PATH_TO_CHECKPOINT>
```