import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
import matplotlib.pyplot as plt  # Libreria aggiunta per i grafici

from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )


    parser.add_argument(
        "--output-plots-dir",
        type=str,
        default="output_plots",
        help="directory where to save the generated histograms",
    )

    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    output_plots = Path(args.output_plots_dir)

    # Crea la cartella di output se non esiste
    output_plots.mkdir(parents=True, exist_ok=True)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    print(f"Generazione grafici in corso nella cartella: {output_plots}")
    
    for txt_file_query in tqdm(txt_files):
        query_name = Path(txt_file_query).stem
        
        # 1. Recupero distanze geometriche
        geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
        
        # 2. Recupero inliers da file torch
        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        query_results = torch.load(torch_file_query, weights_only=False)
        query_db_inliers = torch.zeros(num_preds, dtype=torch.float32)
        for i in range(num_preds):
            query_db_inliers[i] = query_results[i]['num_inliers']
        
        # 3. Ordinamento decrescente in base agli inliers
        query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
        geo_dists = geo_dists[indices]
        
        # 4. Generazione colori condizionali: verde se <= soglia, rosso altrimenti
        colors = ['#2ecc71' if dist <= threshold else '#e74c3c' for dist in geo_dists]

        # 5. Creazione del grafico
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(1, num_preds + 1), query_db_inliers.numpy(), color=colors, edgecolor='grey', alpha=0.8)
        
        # Aggiunta di una legenda personalizzata
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='grey', label=f'Corretto (<= {threshold}m)'),
            Patch(facecolor='#e74c3c', edgecolor='grey', label=f'Errato (> {threshold}m)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Dettagli del grafico
        plt.title(f"Inliers per Predizione - Query {query_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Posizione Predizione (Riordinata)", fontsize=11)
        plt.ylabel("Numero di Inliers", fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xlim(0, num_preds + 1)

        # Salva l'immagine e chiude la figura per liberare memoria
        plot_path = output_plots.joinpath(f"query_{query_name}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    print("Grafici salvati al percorso: {}".format(output_plots))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
