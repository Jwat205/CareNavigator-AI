#!/usr/bin/env python3
"""
Launch a throwaway EC2 instance in the same VPC as the production ECS
service, use it to hit the ALB at high concurrency (proving results aren't
contaminated by the operator's home internet connection - see
../../backend/benchmarks/AWS_LOAD_TEST_RESULTS.md for why that matters),
then terminate itself automatically.

Usage: python run_production_loadtest.py [concurrency ...]
    (default concurrencies: 1 10 50 100 500 1000 1500)

Requires ./.prod_info.json (written by provision.py).
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
INFO_FILE = HERE / ".prod_info.json"
KEY_NAME = "carenav-prod-loadgen-key"


def get_my_ip() -> str:
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as r:
        return r.read().decode().strip()


def main():
    if not INFO_FILE.exists():
        print(f"No {INFO_FILE} - run provision.py first.", file=sys.stderr)
        sys.exit(1)

    concurrencies = sys.argv[1:] or ["1", "10", "50", "100", "500", "1000", "1500"]
    info = json.loads(INFO_FILE.read_text())

    session = boto3.Session(region_name=info["region"])
    ec2 = session.client("ec2")
    ec2r = session.resource("ec2")
    ssm = session.client("ssm")

    my_ip = get_my_ip()
    key_path = HERE / f"{KEY_NAME}.pem"
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
    except ec2.exceptions.ClientError:
        key = ec2.create_key_pair(KeyName=KEY_NAME, KeyType="ed25519")
        with open(key_path, "w", newline="\n") as f:
            f.write(key["KeyMaterial"])
        key_path.chmod(0o400)

    sg_name = "carenav-prod-loadgen-sg"
    existing = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [sg_name]}, {"Name": "vpc-id", "Values": [info["vpc_id"]]}]
    )
    if existing["SecurityGroups"]:
        sg_id = existing["SecurityGroups"][0]["GroupId"]
    else:
        sg = ec2.create_security_group(
            GroupName=sg_name, Description="Ephemeral load-gen box - SSH from operator only", VpcId=info["vpc_id"]
        )
        sg_id = sg["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "SSH from operator"}],
            }],
        )

    ami = ssm.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")[
        "Parameter"
    ]["Value"]

    print("Launching load-gen instance...")
    instances = ec2r.create_instances(
        ImageId=ami,
        InstanceType="t3.medium",
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData="#!/bin/bash\ndnf install -y python3-pip\npip3 install aiohttp\n",
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "carenav-prod-loadgen"}]}],
    )
    instance = instances[0]
    instance.wait_until_running()
    instance.reload()
    ip = instance.public_ip_address
    print(f"Instance {instance.id} at {ip}, waiting for SSH + aiohttp install...")

    ssh = ["ssh", "-i", str(key_path), "-o", "StrictHostKeyChecking=accept-new", f"ec2-user@{ip}"]
    for _ in range(30):
        result = subprocess.run(
            ssh + ["-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "python3 -c 'import aiohttp'"],
            capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(5)
    else:
        print("Instance never became ready; terminating and giving up.", file=sys.stderr)
        instance.terminate()
        sys.exit(1)

    script_path = HERE.parent / "intra_vpc_loadtest.py"
    subprocess.run(
        ["scp", "-i", str(key_path), "-o", "StrictHostKeyChecking=accept-new", str(script_path),
         f"ec2-user@{ip}:/tmp/loadtest.py"],
        check=True,
    )

    target = f"http://{info['alb_dns']}"
    print(f"\nRunning load test against {target} from inside the VPC...\n")
    subprocess.run(ssh + [f"python3 /tmp/loadtest.py {target} {' '.join(concurrencies)}"], check=True)

    print("\nTearing down the load-gen instance...")
    instance.terminate()
    instance.wait_until_terminated()
    try:
        ec2.delete_security_group(GroupId=sg_id)
    except Exception:
        pass
    try:
        ec2.delete_key_pair(KeyName=KEY_NAME)
        key_path.unlink(missing_ok=True)
    except Exception:
        pass
    print("Load-gen instance cleaned up. The production service is still running - "
          "run teardown.py in this directory when you're done with it.")


if __name__ == "__main__":
    main()
