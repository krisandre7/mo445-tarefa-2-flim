
# Pré-requisitos
## Cria ambiente conda:
```bash
mamba create -n flim python=3.11 -y
mamba activate flim
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
pip install -e . --no-build-isolation
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

# Explicação dos Notebooks
- `notebooks/segment_dino.ipynb`: Notebook com a implementação do método alternativo utilizando somente DINOv3, sem FLIM ou IFT.
- `notebooks/tarefa_2_gilson.ipynb`: Notebook com a implementação do método original usando o modelo FLIM. Foi incrementado com a opção de usar DINOv3 no lugar do FLIM. Este é o método (DINOv3 + MLP) discutido no relatório, Para habilitar, modificar USE_FLIM para False.
- `notebooks/experimentos_preliminares.ipynb`: Notebook com experimentos modificanndo apenas o método original, retreinando o FLIM e combinando FLIMs com diferentes profundidades.