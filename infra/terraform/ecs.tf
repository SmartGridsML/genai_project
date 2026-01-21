locals {
  container_secrets = concat(
    [for key, param in aws_ssm_parameter.env : {
      name      = key
      valueFrom = param.arn
    }],
    [for key, secret in aws_secretsmanager_secret.env : {
      name      = key
      valueFrom = secret.arn
    }]
  )

  assign_public_ip = length(var.private_subnet_ids) == 0
}

resource "aws_ecs_cluster" "app" {
  name = "${var.project_name}-cluster"

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = var.log_retention_in_days

  tags = local.tags
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn        = aws_iam_role.ecs_task_execution.arn
  task_role_arn             = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]
      secrets = local.container_secrets
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  tags = local.tags
}

resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.private_subnet_ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = local.assign_public_ip
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "backend"
    container_port   = var.container_port
  }

  depends_on = [
    aws_lb_listener.http
  ]

  tags = local.tags
}
