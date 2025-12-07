#test.py

import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
from model import create_model
import torch.nn.functional as F

# --- Configurações ---
BASE_DIR = 'dataset_waste_container'
OUTPUT_CSV = 'scores_simulados.csv'
MODEL_PATH = '../models/best_model.pth'
NUM_CLASSES = 7

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --------------------------------------------------------------------------

def classifier_score(img_tensor, model, device):

    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    return probs


def run_test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_paths = []

    for root, _, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_paths.append(os.path.join(root, filename))

    if not all_paths:
        print(f"Nenhuma imagem encontrada em '{BASE_DIR}'. Verifique o path.")
        return

    print(f"Total de {len(all_paths)} imagens encontradas.")

    results_list = []

    for path in all_paths:

        # ----- Abrir imagem -----
        img = Image.open(path).convert("RGB")
        img_tensor = test_transform(img)

        # ----- Obter scores -----
        scores = classifier_score(img_tensor, model, device)

        nome_imagem = os.path.basename(path)

        row = {
            'Imagem': nome_imagem,
            'Previsão': np.argmax(scores)
        }

        results_list.append(row)

    # Criar DataFrame
    df = pd.DataFrame(results_list)

    # Exportar
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nResultados exportados para: {OUTPUT_CSV}")
    print("Amostra dos resultados:")
    print(df.head())

