python attack.py mlp-dp nmi \
--dataset facebook \
--shadow_epsilon 8 \
--shadow_num_layers 3 \
--shadow_hidden_dim 64 \
--shadow_activation selu \
--shadow_optimizer adam \
--shadow_learning_rate 0.01 \
--shadow_max_grad_norm 1 \
--shadow_epochs 10 \
--shadow_batch_size 256 \
--shadow_val_interval 0 \
--num_nodes_per_class 1000 \
--attack_hidden_dim 64 \
--attack_num_layers 3 \
--attack_activation selu \
--attack_batch_norm True \
--attack_batch_size full \
--attack_epochs 100 \
--attack_optimizer adam \
--attack_learning_rate 0.01 \
--attack_val_interval 1 \
--repeats 10