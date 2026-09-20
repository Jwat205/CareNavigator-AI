#!/usr/bin/env python3
"""
Stand up a real horizontally-scaled production architecture for
CareNavigator AI on AWS Fargate, to validate "handles 1000+ concurrent"
under an actual load balancer with auto-scaling - not a single box.

Architecture:
    Internet -> ALB (port 80) -> ECS Fargate service (N tasks, port 8000)
    Service auto-scales 2-10 tasks on ALB request count per target.

This is meaningfully different from ../launch_instance.py (which provisions
ONE EC2 box for a quick single-instance test). This script builds the real
multi-instance-behind-a-load-balancer pattern needed to genuinely handle
high concurrency, matching what CareNavigator AI/README.md documents.

Prerequisites:
    - Docker running locally (this script builds and pushes the image)
    - AWS credentials with BOTH deploy/aws/iam-policy.json AND
      deploy/aws/production/iam-policy-production.json attached
    - Run from the repo root's context is NOT required; paths are resolved
      relative to this file

Usage:
    python provision.py [--desired-count 4] [--min-capacity 2] [--max-capacity 10]

Writes ./.prod_info.json for teardown.py to read.
"""
import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
INFO_FILE = HERE / ".prod_info.json"

CLUSTER_NAME = "carenav-prod-cluster"
SERVICE_NAME = "carenav-prod-service"
TASK_FAMILY = "carenav-prod"
ECR_REPO_NAME = "carenavigator-ai"
EXECUTION_ROLE_NAME = "carenav-ecs-task-execution-role"
LOG_GROUP = "/ecs/carenav-prod"
ALB_NAME = "carenav-prod-alb"
TG_NAME = "carenav-prod-tg"
ALB_SG_NAME = "carenav-prod-alb-sg"
TASK_SG_NAME = "carenav-prod-task-sg"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_ecr_repo(ecr):
    try:
        resp = ecr.create_repository(repositoryName=ECR_REPO_NAME)
        return resp["repository"]["repositoryUri"]
    except ecr.exceptions.RepositoryAlreadyExistsException:
        resp = ecr.describe_repositories(repositoryNames=[ECR_REPO_NAME])
        return resp["repositories"][0]["repositoryUri"]


def build_and_push_image(ecr, repo_uri: str) -> str:
    print("Building Docker image from repo root (this takes a few minutes)...")
    local_tag = "carenavigator-ai:prod-latest"
    run(["docker", "build", "-t", local_tag, "."], cwd=str(REPO_ROOT))

    auth = ecr.get_authorization_token()["authorizationData"][0]
    username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":")
    registry = auth["proxyEndpoint"]

    login = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password.encode(),
        check=True,
    )

    remote_tag = f"{repo_uri}:latest"
    run(["docker", "tag", local_tag, remote_tag])
    run(["docker", "push", remote_tag])
    return remote_tag


def ensure_execution_role(iam) -> str:
    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        role = iam.create_role(
            RoleName=EXECUTION_ROLE_NAME,
            AssumeRolePolicyDocument=trust_policy,
            Description="ECS task execution role for CareNavigator AI (pulls from ECR, writes logs)",
        )
        role_arn = role["Role"]["Arn"]
        iam.attach_role_policy(
            RoleName=EXECUTION_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
        )
        print(f"Created execution role {role_arn}, waiting for IAM propagation...")
        time.sleep(10)
        return role_arn
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=EXECUTION_ROLE_NAME)
        return role["Role"]["Arn"]


def ensure_log_group(logs):
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass


def get_default_vpc_and_subnets(ec2):
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    # ALB requires subnets in at least 2 distinct AZs
    seen_az = set()
    subnet_ids = []
    for s in subnets["Subnets"]:
        if s["AvailabilityZone"] not in seen_az:
            seen_az.add(s["AvailabilityZone"])
            subnet_ids.append(s["SubnetId"])
    if len(subnet_ids) < 2:
        raise RuntimeError("Need at least 2 AZs with subnets in the default VPC for an ALB.")
    return vpc_id, subnet_ids


def get_my_ip() -> str:
    import urllib.request
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as r:
        return r.read().decode().strip()


def ensure_security_groups(ec2, vpc_id, my_ip):
    vpc_cidr = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]["CidrBlock"]

    def find_or_create(name, description):
        existing = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )
        if existing["SecurityGroups"]:
            return existing["SecurityGroups"][0]["GroupId"]
        sg = ec2.create_security_group(GroupName=name, Description=description, VpcId=vpc_id)
        return sg["GroupId"]

    alb_sg_id = find_or_create(ALB_SG_NAME, "CareNavigator AI production ALB - HTTP from operator IP")
    task_sg_id = find_or_create(TASK_SG_NAME, "CareNavigator AI production Fargate tasks - only from ALB")

    def authorize(sg_id, permissions):
        try:
            ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=permissions)
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise

    authorize(alb_sg_id, [{
        "IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
        "IpRanges": [
            {"CidrIp": f"{my_ip}/32", "Description": "operator"},
            {"CidrIp": vpc_cidr, "Description": "intra-vpc-loadtest"},
        ],
    }])
    authorize(task_sg_id, [{
        "IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000,
        "UserIdGroupPairs": [{"GroupId": alb_sg_id, "Description": "from ALB only"}],
    }])
    return alb_sg_id, task_sg_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--desired-count", type=int, default=4)
    parser.add_argument("--min-capacity", type=int, default=2)
    parser.add_argument("--max-capacity", type=int, default=10)
    parser.add_argument("--task-cpu", default="1024", help="Fargate task CPU units (1024 = 1 vCPU)")
    parser.add_argument("--task-memory", default="4096", help="Fargate task memory in MB")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    ec2 = session.client("ec2")
    ecr = session.client("ecr")
    ecs = session.client("ecs")
    iam = session.client("iam")
    logs = session.client("logs")
    elbv2 = session.client("elbv2")
    appscaling = session.client("application-autoscaling")

    my_ip = get_my_ip()
    print(f"Your IP: {my_ip} (ALB will only accept HTTP from this address)")

    vpc_id, subnet_ids = get_default_vpc_and_subnets(ec2)
    print(f"Using VPC {vpc_id}, subnets {subnet_ids}")

    alb_sg_id, task_sg_id = ensure_security_groups(ec2, vpc_id, my_ip)
    print(f"Security groups: ALB={alb_sg_id} tasks={task_sg_id}")

    repo_uri = ensure_ecr_repo(ecr)
    image_uri = build_and_push_image(ecr, repo_uri)
    print(f"Image pushed: {image_uri}")

    exec_role_arn = ensure_execution_role(iam)
    ensure_log_group(logs)

    try:
        ecs.create_cluster(clusterName=CLUSTER_NAME)
    except ClientError:
        pass
    print(f"Cluster {CLUSTER_NAME} ready")

    # Run the FastAPI backend only (uvloop auto-detected) - one process per
    # task, scale via replica count. This is the cloud-native pattern:
    # horizontal scaling via task count + ALB, not per-task worker processes.
    task_def = ecs.register_task_definition(
        family=TASK_FAMILY,
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu=args.task_cpu,
        memory=args.task_memory,
        executionRoleArn=exec_role_arn,
        containerDefinitions=[{
            "name": "api",
            "image": image_uri,
            "command": ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
            "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
            "essential": True,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": LOG_GROUP,
                    "awslogs-region": args.region,
                    "awslogs-stream-prefix": "api",
                },
            },
        }],
    )
    task_def_arn = task_def["taskDefinition"]["taskDefinitionArn"]
    print(f"Task definition registered: {task_def_arn}")

    alb = elbv2.create_load_balancer(
        Name=ALB_NAME,
        Subnets=subnet_ids,
        SecurityGroups=[alb_sg_id],
        Scheme="internet-facing",
        Type="application",
        IpAddressType="ipv4",
    )["LoadBalancers"][0]
    alb_arn = alb["LoadBalancerArn"]
    alb_dns = alb["DNSName"]
    print(f"ALB created: {alb_dns}")

    tg = elbv2.create_target_group(
        Name=TG_NAME,
        Protocol="HTTP",
        Port=8000,
        VpcId=vpc_id,
        TargetType="ip",
        HealthCheckPath="/health",
        HealthCheckIntervalSeconds=15,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=3,
    )["TargetGroups"][0]
    tg_arn = tg["TargetGroupArn"]

    elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol="HTTP",
        Port=80,
        DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
    )
    print("Listener created (port 80 -> target group)")

    print("Waiting for ALB to become active...")
    elbv2.get_waiter("load_balancer_available").wait(LoadBalancerArns=[alb_arn])

    ecs.create_service(
        cluster=CLUSTER_NAME,
        serviceName=SERVICE_NAME,
        taskDefinition=task_def_arn,
        desiredCount=args.desired_count,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnet_ids,
                "securityGroups": [task_sg_id],
                "assignPublicIp": "ENABLED",
            }
        },
        loadBalancers=[{"targetGroupArn": tg_arn, "containerName": "api", "containerPort": 8000}],
        healthCheckGracePeriodSeconds=90,
    )
    print(f"Service created, desired count={args.desired_count}. Waiting for it to stabilize...")
    ecs.get_waiter("services_stable").wait(cluster=CLUSTER_NAME, services=[SERVICE_NAME])
    print("Service stable - all tasks passed health checks.")

    resource_id = f"service/{CLUSTER_NAME}/{SERVICE_NAME}"
    appscaling.register_scalable_target(
        ServiceNamespace="ecs",
        ResourceId=resource_id,
        ScalableDimension="ecs:service:DesiredCount",
        MinCapacity=args.min_capacity,
        MaxCapacity=args.max_capacity,
    )
    alb_arn_suffix = "/".join(alb_arn.split("/")[-3:])
    tg_arn_suffix = tg_arn.split(":")[-1]
    appscaling.put_scaling_policy(
        PolicyName="carenav-prod-request-count-tracking",
        ServiceNamespace="ecs",
        ResourceId=resource_id,
        ScalableDimension="ecs:service:DesiredCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "ALBRequestCountPerTarget",
                "ResourceLabel": f"{alb_arn_suffix}/{tg_arn_suffix}",
            },
            "TargetValue": 300.0,
            "ScaleInCooldown": 60,
            "ScaleOutCooldown": 60,
        },
    )
    print(f"Auto-scaling registered: {args.min_capacity}-{args.max_capacity} tasks, "
          f"target 300 req/min per task.")

    info = {
        "region": args.region,
        "cluster": CLUSTER_NAME,
        "service": SERVICE_NAME,
        "task_family": TASK_FAMILY,
        "alb_arn": alb_arn,
        "alb_dns": alb_dns,
        "target_group_arn": tg_arn,
        "alb_security_group_id": alb_sg_id,
        "task_security_group_id": task_sg_id,
        "ecr_repo_name": ECR_REPO_NAME,
        "execution_role_name": EXECUTION_ROLE_NAME,
        "log_group": LOG_GROUP,
        "vpc_id": vpc_id,
    }
    INFO_FILE.write_text(json.dumps(info, indent=2))

    print("\nProduction stack is up.")
    print(json.dumps(info, indent=2))
    print(f"\nBackend URL: http://{alb_dns}")
    print("Give DNS ~30-60s to propagate, then test:")
    print(f"  curl http://{alb_dns}/health")
    print("\nWhen you're done: python teardown.py")


if __name__ == "__main__":
    main()
