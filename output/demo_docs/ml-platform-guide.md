# Internal ML Platform Guide
Model training jobs run on the Kubeflow cluster; request quota through the ml-infra portal.
Every production model must be registered in the model registry with a model card covering
intended use, training data lineage, evaluation metrics, and known limitations. Models serving
customer traffic require shadow evaluation for one week and sign-off from the ML review board.
Feature pipelines are defined in the feature store; ad-hoc features computed in notebooks may
not ship to production. Drift monitoring alerts when population stability index exceeds 0.2,
triggering scheduled retraining. GPU quota resets monthly and unused quota does not roll over.
