"""polymer_ranking
Polymer Carrier Mobility Prediction 
================================================
Usage:
# random split: pairs are randomly split into train/val/test sets;
# the model is saved as checkpoints/best_model.safetensors
# group split by paper (doi): pairs from the same paper never cross train/val/test;
# the model is saved as checkpoints/best_model_group.safetensors

python polymer_ranking.py --mode train --csv contrastive_full_paired.csv

python polymer_ranking.py --mode finetune --csv contrastive_full_paired.csv --checkpoint checkpoints/best_model.safetensors

python polymer_ranking.py --mode predict --predict_csv new_mol.csv --checkpoint checkpoints/final_model.safetensors

python polymer_ranking.py --mode train --csv contrastive_full_paired.csv --split_method group

python polymer_ranking.py --mode predict --predict_csv new_mol.csv --checkpoint checkpoints/best_model_group.safetensors

# hyperparam search

python hyperparam_search.py --stage all --n_trials 10 --max_epochs 40

"""

from polymer_ranking.cli import main

if __name__ == "__main__":
    main()
