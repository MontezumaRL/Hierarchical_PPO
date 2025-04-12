.PHONY: env ppo

ENV_DIR := venv

env:
	python -m venv $(ENV_DIR)
	$(ENV_DIR)/Scripts/pip install -r requirements.txt
	@echo "Environnement créé et packages installés."
	@echo "Pour l'activer :"
	@echo "    call $(ENV_DIR)\Scripts\activate"  # pour Windows
	@echo "    source $(ENV_DIR)/bin/activate"    # pour Linux/macOS

ppo:
	python ppo.py --num_epochs=$(num_epochs) --policy_model=$(policy_model) --value_model=$(value_model) --save_path_policy=$(save_path_policy) --save_path_value=$(save_path_value)
