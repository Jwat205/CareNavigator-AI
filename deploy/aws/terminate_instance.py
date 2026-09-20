#!/usr/bin/env python3
"""
Terminate the load-test instance launch_instance.py created, and clean up
the security group. Leaves the local key pair file alone (harmless, free)
but you can delete .instance_info.json afterward if you want a clean slate.

Usage: python terminate_instance.py
"""
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
INFO_FILE = HERE / ".instance_info.json"


def main():
    if not INFO_FILE.exists():
        print(f"No {INFO_FILE} found — nothing to terminate.", file=sys.stderr)
        sys.exit(1)

    info = json.loads(INFO_FILE.read_text())
    session = boto3.Session(region_name=info["region"])
    ec2_client = session.client("ec2")

    print(f"Terminating instance {info['instance_id']}...")
    ec2_client.terminate_instances(InstanceIds=[info["instance_id"]])
    waiter = ec2_client.get_waiter("instance_terminated")
    waiter.wait(InstanceIds=[info["instance_id"]])
    print("Instance terminated.")

    sg_id = info.get("security_group_id")
    if sg_id:
        # The security group can only be deleted once the instance is fully
        # gone (ENI detachment can lag slightly behind "terminated").
        for attempt in range(6):
            try:
                ec2_client.delete_security_group(GroupId=sg_id)
                print(f"Deleted security group {sg_id}.")
                break
            except ClientError as e:
                if attempt == 5:
                    print(
                        f"Could not delete security group {sg_id} ({e}); "
                        "it's free to leave behind, or delete it manually later.",
                        file=sys.stderr,
                    )
                else:
                    time.sleep(10)

    INFO_FILE.unlink()
    print("Done. Verify in the AWS console (EC2 > Instances) that nothing is still running.")


if __name__ == "__main__":
    main()
