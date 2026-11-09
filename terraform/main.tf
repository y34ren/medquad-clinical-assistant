# Declarative Terraform Infrastructure-as-Code (IaC) for NIH MedQuAD Clinical Assistant
# Provisions Cloud Run v2, Vertex AI Search, Cloud SQL, BigQuery Audit Sink, Secret Manager, and Least-Privilege IAM

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Dedicated Least-Privilege Agent Service Account (Day 1 Agent Identity)
resource "google_service_account" "medquad_agent_sa" {
  account_id   = "medquad-agent-sa"
  display_name = "NIH MedQuAD Clinical Assistant Runtime Service Account"
  description  = "Scoped principal for RootOrchestrator, Vertex AI Gemini 2.5, MedQuAD Search, and Secret Manager."
}

# 2. Least-Privilege IAM Role Bindings
locals {
  agent_iam_roles = [
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/secretmanager.secretAccessor",
    "roles/cloudsql.client",
    "roles/bigquery.dataEditor",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "medquad_agent_permissions" {
  for_each = toset(local.agent_iam_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.medquad_agent_sa.email}"
}

# 3. Google Cloud Secret Manager Secrets (Rubric 5.2)
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "GEMINI_API_KEY"
  replication {
    auto {}
  }
  labels = {
    system     = "medquad-clinical-assistant"
    compliance = "hipaa-baa"
  }
}

resource "google_secret_manager_secret" "medquad_db_uri" {
  secret_id = "MEDQUAD_DB_URI"
  replication {
    auto {}
  }
  labels = {
    system     = "medquad-clinical-assistant"
    compliance = "hipaa-baa"
  }
}

# 4. Vertex AI Search Data Store for NIH MedQuAD Corpus Grounding
resource "google_discovery_engine_data_store" "medquad_corpus" {
  location                    = "global"
  data_store_id               = "nih-medquad-clinical-corpus-${var.environment}"
  display_name                = "NIH MedQuAD Grounded Clinical Q&A & Contraindication Datastore"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false
}

# 5. Cloud SQL PostgreSQL Instance for Persistent Session & HITL Escalation State
resource "google_sql_database_instance" "clinical_session_db" {
  name                = "nih-medquad-session-db-${var.environment}"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = true

  settings {
    tier              = "db-custom-2-7680"
    availability_type = "REGIONAL"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled = false
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }
}

# 6. BigQuery Dataset & Table for PHI-Scrubbed Clinical Audit & OpenTelemetry Spans
resource "google_bigquery_dataset" "clinical_audit_logs" {
  dataset_id  = "nih_medquad_clinical_audit_${var.environment}"
  description = "HIPAA-compliant PHI-scrubbed clinical telemetry, HITL sign-offs, and trajectory audit logs."
  location    = var.region
}

resource "google_bigquery_table" "clinical_telemetry_spans" {
  dataset_id          = google_bigquery_dataset.clinical_audit_logs.dataset_id
  table_id            = "opentelemetry_clinical_spans"
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
}

# 7. Cloud Run v2 Service Hosting the FastAPI + Google ADK Agent Gateway
resource "google_cloud_run_v2_service" "medquad_assistant" {
  name     = "medquad-clinical-assistant"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.medquad_agent_sa.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "MEDQUAD_DB_URI"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.medquad_db_uri.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}

output "cloud_run_service_uri" {
  description = "Internal HTTPS endpoint of the NIH MedQuAD Clinical Assistant."
  value       = google_cloud_run_v2_service.medquad_assistant.uri
}

output "a2a_agent_card_url" {
  description = "Day 5 A2A Protocol Agent Card discovery endpoint."
  value       = "${google_cloud_run_v2_service.medquad_assistant.uri}/.well-known/agent.json"
}
