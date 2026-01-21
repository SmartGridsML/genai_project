output "alb_dns_name" {
  description = "Public DNS name for the application load balancer."
  value       = aws_lb.app.dns_name
}

output "s3_bucket_name" {
  description = "S3 bucket for generated documents."
  value       = aws_s3_bucket.documents.bucket
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.app.name
}

output "prometheus_workspace_id" {
  description = "Amazon Managed Prometheus workspace ID."
  value       = aws_prometheus_workspace.main.id
}

output "grafana_workspace_endpoint" {
  description = "Amazon Managed Grafana endpoint."
  value       = aws_grafana_workspace.main.endpoint
}
