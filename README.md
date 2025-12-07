# Container Classification

Este repositório contém o projeto desenvolvido no âmbito da UC **Processamento de Dados Audiovisuais**, cujo objetivo é treinar um sistema de classificação automática de imagens capaz de reconhecer **7 tipos de contentores de lixo urbanos**.

O sistema foi construído com **PyTorch**, usando **transfer learning** com a arquitetura **ResNet18**, complementada com várias técnicas de pré-processamento, aumento de dados e regularização.

---
## Requisitos e Instalação

O projeto requer as seguintes bibliotecas Python, que podem ser instaladas através do `pip` com o ficheiro  `requirements.txt`:

```bash
pip install -r requirements.txt
```

##  Estrutura do Repositório
```
├── dataset_waste_container
│ ├── container_battery
│ ├── container_biodegradable
│ ├── container_blue
│ ├── container_default
│ ├── container_green
│ ├── container_oil
│ └── container_yellow
│
├── models/
│ └── best_model.pth # modelo final treinado
│
├── src/
│ ├── main.py # pipeline principal (treino + avaliação)
│ ├── train.py # função de treino 
│ ├── evaluate.py # métricas + matriz de confusão
│ ├── model.py # criação do modelo
│ ├── data_preparation.py # dataloaders + transforms
│ └── test.py # script para gerar previsões em lote
│
├── requirements.txt
│
└── README.md
```



