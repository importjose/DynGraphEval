MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db

mlflow:
	MLFLOW_TRACKING_URI=$(MLFLOW_DB) mlflow ui
