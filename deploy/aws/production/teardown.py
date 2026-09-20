#!/usr/bin/env python3
"""
Tear down everything provision.py created, in dependency order, so nothing
keeps billing after you're done testing. Leaves the ECS task execution IAM
role in place (it costs nothing and is safe to reuse next time) unless you
pass --delete-role.

Usage: python teardown.py [--delete-role] [--delete-ecr-repo]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
INFO_FILE = HERE / ".prod_info.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete-role", action="store_true", help="Also delete the IAM execution role")
    parser.add_argument("--delete-ecr-repo", action="store_true", help="Also delete the ECR repo and all images")
    args = parser.parse_args()

    if not INFO_FILE.exists():
        print(f"No {INFO_FILE} found - nothing to tear down.", file=sys.stderr)
        sys.exit(1)

    info = json.loads(INFO_FILE.read_text())
    session = boto3.Session(region_name=info["region"])
    ecs = session.client("ecs")
    elbv2 = session.client("elbv2")
    appscaling = session.client("application-autoscaling")
    ec2 = session.client("ec2")
    logs = session.client("logs")
    iam = session.client("iam")
    ecr = session.client("ecr")

    resource_id = f"service/{info['cluster']}/{info['service']}"

    print("Removing auto-scaling policy and scalable target...")
    try:
        appscaling.delete_scaling_policy(
            PolicyName="carenav-prod-request-count-tracking",
            ServiceNamespace="ecs",
            ResourceId=resource_id,
            ScalableDimension="ecs:service:DesiredCount",
        )
    except ClientError as e:
        print(f"  (skip: {e})")
    try:
        appscaling.deregister_scalable_target(
            ServiceNamespace="ecs", ResourceId=resource_id, ScalableDimension="ecs:service:DesiredCount"
        )
    except ClientError as e:
        print(f"  (skip: {e})")

    print(f"Scaling service {info['service']} to 0 and deleting...")
    try:
        ecs.update_service(cluster=info["cluster"], service=info["service"], desiredCount=0)
        ecs.get_waiter("services_stable").wait(cluster=info["cluster"], services=[info["service"]])
        ecs.delete_service(cluster=info["cluster"], service=info["service"])
    except ClientError as e:
        print(f"  (skip: {e})")

    print("Deleting listener/load balancer/target group...")
    try:
        listeners = elbv2.describe_listeners(LoadBalancerArn=info["alb_arn"])["Listeners"]
        for l in listeners:
            elbv2.delete_listener(ListenerArn=l["ListenerArn"])
        elbv2.delete_load_balancer(LoadBalancerArn=info["alb_arn"])
        elbv2.get_waiter("load_balancers_deleted").wait(LoadBalancerArns=[info["alb_arn"]])
    except ClientError as e:
        print(f"  (skip: {e})")
    try:
        elbv2.delete_target_group(TargetGroupArn=info["target_group_arn"])
    except ClientError as e:
        print(f"  (skip: {e})")

    print(f"Deregistering task definition family {info['task_family']}...")
    try:
        revisions = ecs.list_task_definitions(familyPrefix=info["task_family"])["taskDefinitionArns"]
        for arn in revisions:
            ecs.deregister_task_definition(taskDefinition=arn)
    except ClientError as e:
        print(f"  (skip: {e})")

    print(f"Deleting cluster {info['cluster']}...")
    try:
        ecs.delete_cluster(cluster=info["cluster"])
    except ClientError as e:
        print(f"  (skip: {e})")

    print("Deleting security groups (retrying - ENI detachment can lag)...")
    for sg_id in [info["task_security_group_id"], info["alb_security_group_id"]]:
        for attempt in range(6):
            try:
                ec2.delete_security_group(GroupId=sg_id)
                print(f"  deleted {sg_id}")
                break
            except ClientError as e:
                if attempt == 5:
                    print(f"  could not delete {sg_id} ({e}); delete manually later, it's free to leave")
                else:
                    time.sleep(10)

    print(f"Deleting log group {info['log_group']}...")
    try:
        logs.delete_log_group(logGroupName=info["log_group"])
    except ClientError as e:
        print(f"  (skip: {e})")

    if args.delete_ecr_repo:
        print(f"Deleting ECR repo {info['ecr_repo_name']} and all images...")
        try:
            ecr.delete_repository(repositoryName=info["ecr_repo_name"], force=True)
        except ClientError as e:
            print(f"  (skip: {e})")
    else:
        print(f"Leaving ECR repo {info['ecr_repo_name']} in place (small storage cost; "
              f"re-run with --delete-ecr-repo to remove it, or reuse it next deploy).")

    if args.delete_role:
        print(f"Deleting IAM role {info['execution_role_name']}...")
        try:
            iam.detach_role_policy(
                RoleName=info["execution_role_name"],
                PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
            )
            iam.delete_role(RoleName=info["execution_role_name"])
        except ClientError as e:
            print(f"  (skip: {e})")
    else:
        print(f"Leaving IAM role {info['execution_role_name']} in place (free; "
              f"re-run with --delete-role to remove it).")

    INFO_FILE.unlink()
    print("\nDone. Verify in the AWS console: EC2 > Load Balancers, ECS > Clusters, "
          "EC2 > Security Groups - confirm nothing named carenav-prod-* remains running.")


if __name__ == "__main__":
    main()
