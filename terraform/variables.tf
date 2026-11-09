variable "project_id" {
  description = "Google Cloud Project ID for the NIH Clinical Center environment."
  type        = string
  default     = "nih-clinical-center-prod"
}

variable "region" {
  description = "Primary GCP deployment region (HIPAA BAA compliant)."
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Artifact Registry container image URI for the MedQuAD Clinical Assistant."
  type        = string
  default     = "us-central1-docker.pkg.dev/nih-clinical-center-prod/nih-clinical-agents/medquad-clinical-assistant:latest"
}

variable "environment" {
  description = "Deployment environment tier (prod, staging, dev)."
  type        = string
  default     = "prod"
}
