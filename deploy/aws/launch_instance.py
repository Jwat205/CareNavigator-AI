#!/usr/bin/env python3
"""
Launch a single EC2 instance for a short-lived load test of CareNavigator AI.

Does NOT build or run the app itself — it only provisions the box (key pair,
security group scoped to your current public IP, instance with Docker
pre-installed via user-data). Once it's running, use deploy_app.sh to ship
the code over and start the container.

Requires credentials to already be available via one of the normal boto3
methods (env vars AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN,
an AWS_PROFILE, or ~/.aws/credentials) — this script never asks for or
accepts credentials as arguments.

Usage:
    python launch_instance.py [--instance-type m6i.xlarge] [--region us-east-1]

Writes ./.instance_info.json with the instance id, public IP, key path, and
security group id — deploy_app.sh and terminate_instance.py both read it.
"""
import argparse
import ipaddress
import json
import sys
import time
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
INFO_FILE = HERE / ".instance_info.json"
KEY_NAME = "carenav-loadtest-key"
SG_NAME = "carenav-loadtest-sg"
USER_DATA = """#!/bin/bash
set -e
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
"""


def get_my_public_ip() -> str:
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as r:
        ip = r.read().decode().strip()
    ipaddress.ip_address(ip)  # validate
    return ip


def get_latest_al2023_ami(ssm) -> str:
    param = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return param["Parameter"]["Value"]


def ensure_key_pair(ec2, ec2_client) -> Path:
    key_path = HERE / f"{KEY_NAME}.pem"
    try:
        ec2_client.describe_key_pairs(KeyNames=[KEY_NAME])
        if not key_path.exists():
            print(
                f"WARNING: key pair '{KEY_NAME}' already exists in AWS but "
                f"{key_path} is missing locally — you won't be able to SSH in. "
                "Delete the key pair in AWS and re-run, or point at the "
                "correct .pem file yourself.",
                file=sys.stderr,
            )
        return key_path
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidKeyPair.NotFound":
            raise

    key = ec2_client.create_key_pair(KeyName=KEY_NAME, KeyType="ed25519")
    # newline="\n" is required on Windows: the default text-mode write
    # translates \n -> \r\n, which corrupts the PEM structure and makes
    # OpenSSH/libcrypto reject the key ("error in libcrypto").
    with open(key_path, "w", newline="\n") as f:
        f.write(key["KeyMaterial"])
    key_path.chmod(0o400)
    print(f"Created key pair, saved private key to {key_path}")
    return key_path


def ensure_security_group(ec2_client, my_ip: str) -> str:
    try:
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"Reusing existing security group {sg_id}")
            return sg_id
    except ClientError:
        pass

    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    sg = ec2_client.create_security_group(
        GroupName=SG_NAME,
        Description="Short-lived CareNavigator AI load test box - SSH + app ports from your IP only",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]

    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "SSH from operator"}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "FastAPI backend from operator"}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 8501,
                "ToPort": 8501,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "Streamlit from operator"}],
            },
        ],
    )
    print(f"Created security group {sg_id}, locked to {my_ip}/32")
    return sg_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-type", default="m6i.xlarge")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    ec2_client = session.client("ec2")
    ec2 = session.resource("ec2")
    ssm = session.client("ssm")

    my_ip = get_my_public_ip()
    print(f"Your public IP: {my_ip} (only this IP will be allowed to reach the instance)")

    ami_id = get_latest_al2023_ami(ssm)
    key_path = ensure_key_pair(ec2, ec2_client)
    sg_id = ensure_security_group(ec2_client, my_ip)

    print(f"Launching {args.instance_type} in {args.region} from AMI {ami_id}...")
    instances = ec2.create_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=USER_DATA,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "carenav-loadtest"}],
            }
        ],
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 30, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
    )
    instance = instances[0]
    print(f"Instance {instance.id} launching, waiting for it to enter 'running' state...")
    instance.wait_until_running()
    instance.reload()

    print("Waiting for status checks to pass (this also means Docker is likely ready)...")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance.id])

    info = {
        "instance_id": instance.id,
        "public_ip": instance.public_ip_address,
        "region": args.region,
        "key_path": str(key_path),
        "security_group_id": sg_id,
    }
    INFO_FILE.write_text(json.dumps(info, indent=2))

    print("\nInstance is up.")
    print(json.dumps(info, indent=2))
    print(f"\nNext: ./deploy_app.sh   (reads {INFO_FILE.name} automatically)")


if __name__ == "__main__":
    main()
