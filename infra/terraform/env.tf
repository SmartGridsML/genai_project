resource "aws_ssm_parameter" "env" {
  for_each = var.environment

  name  = "/${var.project_name}/${each.key}"
  type  = "String"
  value = each.value

  tags = local.tags
}

resource "aws_secretsmanager_secret" "env" {
  for_each = var.secrets

  name = "${var.project_name}/${each.key}"

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "env" {
  for_each = var.secrets

  secret_id     = aws_secretsmanager_secret.env[each.key].id
  secret_string = each.value
}
