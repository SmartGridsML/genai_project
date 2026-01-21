variable "project_name" {
  type        = string
  description = "Project identifier used for resource naming."
  default     = "genai-project"
}

variable "aws_region" {
  type        = string
  description = "AWS region for deployment."
  default     = "us-east-1"
}

variable "vpc_id" {
  type        = string
  description = "Existing VPC ID. Leave empty to use the default VPC."
  default     = ""
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnet IDs for the ALB. Leave empty to use VPC subnets."
  default     = []
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for ECS tasks. Leave empty to use VPC subnets."
  default     = []
}

variable "container_image" {
  type        = string
  description = "Container image for the backend service."
}

variable "container_port" {
  type        = number
  description = "Container port exposed by the backend service."
  default     = 8000
}

variable "desired_count" {
  type        = number
  description = "Desired number of ECS tasks."
  default     = 2
}

variable "cpu" {
  type        = number
  description = "CPU units for the task definition."
  default     = 512
}

variable "memory" {
  type        = number
  description = "Memory (MiB) for the task definition."
  default     = 1024
}

variable "log_retention_in_days" {
  type        = number
  description = "CloudWatch log retention in days."
  default     = 14
}

variable "health_check_path" {
  type        = string
  description = "Health check path for the ALB target group."
  default     = "/health"
}

variable "s3_bucket_name" {
  type        = string
  description = "S3 bucket name for generated documents."
}

variable "environment" {
  type        = map(string)
  description = "Non-secret environment variables stored in SSM Parameter Store."
  default     = {}
}

variable "secrets" {
  type        = map(string)
  description = "Secret environment variables stored in Secrets Manager."
  default     = {}
}

variable "alarm_sns_topic_arn" {
  type        = string
  description = "SNS topic ARN for alarm notifications. Leave empty to disable actions."
  default     = ""
}
