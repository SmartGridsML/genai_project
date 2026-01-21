locals {
  vpc_id             = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default.id
  public_subnet_ids  = length(var.public_subnet_ids) > 0 ? var.public_subnet_ids : data.aws_subnets.default.ids
  private_subnet_ids = length(var.private_subnet_ids) > 0 ? var.private_subnet_ids : data.aws_subnets.default.ids

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = {
    Project = var.project_name
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
}
