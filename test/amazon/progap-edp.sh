python train.py progap-edp --dataset amazon --phases 6 --epsilon 4 --base_layers 2 --head_layers 1 --jk cat --hidden_dim 16 --activation selu --optimizer adam --learning_rate 0.01 --repeats 1 --batch_norm True --epochs 100 --batch_size full $@