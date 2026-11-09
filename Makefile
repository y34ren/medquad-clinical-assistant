.PHONY: test eval run docker-build terraform-check deploy clean

PYTHON := python3

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

eval:
	$(PYTHON) eval/eval_harness.py

run:
	uvicorn app:app --host 0.0.0.0 --port 8080 --reload

docker-build:
	docker build -t medquad-clinical-assistant:latest .

terraform-check:
	cd terraform && terraform fmt -check && terraform validate

deploy:
	bash ./deploy_gcloud.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/clinical_memory.sqlite3
