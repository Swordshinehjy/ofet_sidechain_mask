"""polymer_ranking
Polymer Carrier Mobility Prediction 
================================================
Usage:

python polymer_ranking.py --mode train --csv contrastive_full_paired.csv

python polymer_ranking.py --mode finetune --csv contrastive_full_paired.csv --checkpoint checkpoints/best_model.pt

python polymer_ranking.py --mode predict --predict_csv new_mol.csv --checkpoint checkpoints/final_model.pt

"""

from polymer_ranking.cli import main

if __name__ == "__main__":
    main()
